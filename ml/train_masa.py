"""
Entrena un modelo PyTorch que genera la MASA NORMATIVA óptima a partir de la
geometría del lote y la edificabilidad de Cali (IO del uso + IC/pisos del POT).

Idea (aprendizaje supervisado sobre un óptimo conocido):
  - Se generan miles de lotes sintéticos con geometría realista (frente, fondo,
    área, compacidad) y normas reales de Cali (IO del uso, IC tope del POT).
  - Para cada lote se busca por optimización la masa que MAXIMIZA el área
    construible respetando: ocupación (IO), edificabilidad (IC), geometría y
    aislamiento posterior que crece con la altura.
  - Un MLP aprende:  features(geometría + norma)  ->  parámetros de masa
    (uso de frente, uso de fondo, nº de pisos, retiro).
  El resultado es un "optimizador aprendido": inferencia instantánea de una
  masa normativa que aprovecha el lote, sin correr la búsqueda en el navegador.

Salida: ml/masa_model.json  (pesos + normalización, para el forward-pass en JS).
"""
import json, math, os, numpy as np
import torch, torch.nn as nn

torch.manual_seed(7); np.random.seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))

# ----- normalización de features / targets -----
FEAT_SCALE = np.array([1000., 50., 50., 3., 1., 1., 8., 20.])  # area,front,depth,aspect,compac,io,icMax,maxFloors
FLOOR_CAP  = 20.0
SETBACK_CAP= 5.0

def make_lot():
    """Geometría de lote realista (m)."""
    front = np.random.uniform(5, 45)
    depth = np.random.uniform(8, 60)
    # compacidad: 1=rectángulo perfecto; lotes irregulares aprovechan menos
    compac = np.random.uniform(0.6, 1.0)
    area = front * depth * compac * np.random.uniform(0.9, 1.0)
    aspect = front / depth
    return front, depth, area, aspect, compac

def make_norm():
    """Norma de Cali: IO del uso e IC/pisos del POT (según distribución real)."""
    io = np.random.choice([0.40,0.50,0.60,0.70,0.75,0.80,0.85])
    if np.random.rand() < 0.35:
        # modo pisos (ej. "DOS (2) PISOS"): IC efectivo = pisos * io
        pisos = np.random.randint(2, 9)
        ic = pisos * io
        maxFloors = pisos
    else:
        # modo índice (ej. ICB 3,5 + ICA 4 = 7,5)
        ic = np.random.uniform(0.8, 8.0)
        maxFloors = 20
    return io, ic, maxFloors

# retículas de búsqueda del óptimo (etiqueta)
FU = np.linspace(0.5, 1.0, 6)
DU = np.linspace(0.4, 1.0, 6)
SS = np.array([0., 1.5, 3.0])

def optimal_mass(front, depth, area, compac, io, ic, maxFloors):
    """Busca (frontUse, depthUse, floors, setback) que maximiza el área construible."""
    usable = compac * area            # área realmente edificable del lote
    footCap = min(io*area, usable)    # tope de ocupación en primer piso
    builtCap= ic*area                 # tope de edificabilidad (IC)
    best = None
    for s in SS:
        for fu in FU:
            fL = fu*front
            for du in DU:
                for n in range(1, int(maxFloors)+1):
                    # aislamiento posterior crece con la altura -> reduce fondo útil
                    dEff = du*max(depth - s - 0.25*(n-1), 1.0)
                    foot = fL*dEff
                    if foot <= 0 or foot > footCap+1e-6:
                        continue
                    built = foot*n
                    if built > builtCap+1e-6:
                        continue
                    # objetivo: maximizar construido; desempate: menos pisos y retiro
                    score = built - 0.002*n - 0.001*s
                    if best is None or score > best[0]:
                        best = (score, fu, du, n, s, built)
    if best is None:
        return 0.5, 0.5, 1, 0.0, 0.0
    _, fu, du, n, s, built = best
    return fu, du, n, s, built

def build_dataset(N):
    X, Y = [], []
    for _ in range(N):
        front, depth, area, aspect, compac = make_lot()
        io, ic, maxFloors = make_norm()
        fu, du, n, s, built = optimal_mass(front, depth, area, compac, io, ic, maxFloors)
        X.append([area, front, depth, aspect, compac, io, ic, maxFloors])
        Y.append([fu, du, n/FLOOR_CAP, s/SETBACK_CAP])
    X = (np.array(X, np.float64) / FEAT_SCALE).astype(np.float32)
    Y = np.array(Y, np.float32)
    return torch.tensor(X), torch.tensor(Y)

class MasaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,32), nn.ReLU(),
            nn.Linear(32,32), nn.ReLU(),
            nn.Linear(32,4), nn.Sigmoid())
    def forward(self, x): return self.net(x)

def main():
    print("Generando datos sintéticos (geometría × norma Cali)…")
    Xtr, Ytr = build_dataset(9000)
    Xva, Yva = build_dataset(1500)
    model = MasaNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.MSELoss()
    print("Entrenando…")
    for ep in range(400):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xtr), Ytr); loss.backward(); opt.step()
        if (ep+1) % 50 == 0:
            model.eval()
            with torch.no_grad(): vl = lossf(model(Xva), Yva).item()
            print(f"  época {ep+1:3d}  train {loss.item():.4f}  val {vl:.4f}")

    # métrica útil: ¿qué % del aprovechamiento óptimo logra el modelo?
    model.eval()
    with torch.no_grad(): P = model(Xva).numpy()
    Xr = Xva.numpy()*FEAT_SCALE; ratios=[]
    for i in range(len(Xr)):
        area,front,depth,aspect,compac,io,ic,mf = Xr[i]
        fu,du,nn_,s = P[i]; n=max(1,round(nn_*FLOOR_CAP)); s=s*SETBACK_CAP
        usable=compac*area; footCap=min(io*area,usable); builtCap=ic*area
        dEff=du*max(depth - s - 0.25*(n-1),1.0); foot=min(fu*front*dEff, footCap)
        built=min(foot*n, builtCap) if foot>0 else 0
        _,_,_,_,bopt = optimal_mass(front,depth,area,compac,io,ic,mf)
        if bopt>0: ratios.append(built/bopt)
    ratios=np.array(ratios)
    print(f"Aprovechamiento ML vs óptimo: media {ratios.mean()*100:.1f}%  (p10 {np.percentile(ratios,10)*100:.1f}%)")

    # exportar pesos + normalización a JSON (forward-pass en el navegador)
    sd = model.state_dict()
    def lw(k): return sd[k].numpy().tolist()
    out = {
        "arch":[8,32,32,4], "act":["relu","relu","sigmoid"],
        "featScale": FEAT_SCALE.tolist(), "floorCap":FLOOR_CAP, "setbackCap":SETBACK_CAP,
        "layers":[
            {"W":lw("net.0.weight"),"b":lw("net.0.bias")},
            {"W":lw("net.2.weight"),"b":lw("net.2.bias")},
            {"W":lw("net.4.weight"),"b":lw("net.4.bias")},
        ],
        "trainedOn": 9000, "note":"MasaNet: geometría+norma Cali -> (frontUse,depthUse,floors,setback)"
    }
    path = os.path.join(HERE, "masa_model.json")
    with open(path,"w",encoding="utf-8") as f: json.dump(out,f)
    print("Pesos exportados ->", path, f"({os.path.getsize(path)//1024} KB)")

if __name__ == "__main__":
    main()
