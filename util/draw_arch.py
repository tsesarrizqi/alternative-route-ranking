"""
Generate a clean architecture diagram as PNG.
Run: python util/draw_arch.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(20, 9))
ax.set_xlim(0, 20)
ax.set_ylim(0, 9)
ax.axis("off")

C_NODE = "#D6EAF8"; C_FWD = "#D5F5E3"; C_BWD = "#FDEBD0"
C_MLP  = "#E8DAEF"; C_DIV = "#F2F3F4"; C_OUT = "#EBF5FB"
C_UBS  = "#FFFDE7"; C_POOL= "#FDEDEC"; C_SPAN= "#FADBD8"
BORDER = "#444444"; GRN = "#1E8449"; ORG = "#A04000"
RED = "#922B21"; GOLD = "#9A7D0A"

def rbox(x,y,w,h,color,label,fontsize=9,bold=False,border=BORDER,lw=1.4,ls="-"):
    r=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.07",linewidth=lw,
                     edgecolor=border,facecolor=color,zorder=3,linestyle=ls)
    ax.add_patch(r)
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fontsize,
            fontweight="bold" if bold else "normal",zorder=4,multialignment="center")

def arr(x0,y0,x1,y1,color=BORDER,lw=1.5,ls="-"):
    ax.annotate("",xy=(x1,y1),xytext=(x0,y0),
                arrowprops=dict(arrowstyle="-|>",color=color,lw=lw,linestyle=ls),zorder=5)

def darr(x0,y0,x1,y1,c="#AAAAAA"):
    ax.plot([x0,x1],[y0,y1],color=c,lw=1.0,linestyle=":",zorder=2)

#  Left panel title
ax.text(3.7,8.75,"How BiGRU reasons about any subpath (vi -> vj)",
        ha="center",fontsize=11,fontweight="bold")

NW,NH=0.68,0.52
xs =[0.25,1.05,1.85,2.65,3.45,4.25,5.05,5.85]
nms=["v1","...","vi","vi+1","...","vj","...","vn"]

for k,(lbl,nx) in enumerate(zip(nms,xs)):
    hi=lbl in("vi","vj")
    rbox(nx,7.85,NW,NH,C_SPAN if hi else C_NODE,lbl,fontsize=9,
         border=RED if hi else BORDER,lw=2.0 if hi else 1.2)
    if k<len(xs)-1 and lbl!="..." and nms[k+1]!="...":
        arr(nx+NW,7.85+NH/2,xs[k+1],7.85+NH/2,color="#888888",lw=1.0)

ax.annotate("",xy=(xs[5]+NW,7.72),xytext=(xs[2],7.72),
            arrowprops=dict(arrowstyle="<->",color=RED,lw=1.6),zorder=6)
ax.text((xs[2]+xs[5]+NW)/2,7.50,"subpath",ha="center",fontsize=8,color=RED)

fl=["h->1","...","h->i","h->i+1","...","h->j","...","h->n"]
for k,(lbl,nx) in enumerate(zip(fl,xs)):
    hi=lbl=="h->j"
    rbox(nx,6.70,NW,NH,"#A9DFBF" if hi else C_FWD,lbl,fontsize=8,
         border=GRN,lw=2.0 if hi else 1.2)
    if k<len(xs)-1 and lbl!="..." and fl[k+1]!="...":
        arr(nx+NW,6.70+NH/2,xs[k+1],6.70+NH/2,color=GRN,lw=1.2)
ax.text(xs[0]-0.22,6.70+NH/2,"Fwd ->",ha="right",va="center",
        fontsize=8,color=GRN,fontweight="bold")

bl=["h<-1","...","h<-i","h<-i+1","...","h<-j","...","h<-n"]
for k,(lbl,nx) in enumerate(zip(bl,xs)):
    hi=lbl=="h<-i"
    rbox(nx,5.90,NW,NH,"#F0B27A" if hi else C_BWD,lbl,fontsize=8,
         border=ORG,lw=2.0 if hi else 1.2)
    if k>0 and lbl!="..." and bl[k-1]!="...":
        arr(nx,5.90+NH/2,xs[k-1]+NW,5.90+NH/2,color=ORG,lw=1.2)
ax.text(xs[0]-0.22,5.90+NH/2,"<- Bwd",ha="right",va="center",
        fontsize=8,color=ORG,fontweight="bold")

arr(xs[5]+NW/2,6.70,xs[5]+NW/2,6.33,color=GRN,lw=1.2)
arr(xs[2]+NW/2,5.90,xs[2]+NW/2,5.53,color=ORG,lw=1.2)

AW=1.65; AH=0.72
AX1=xs[2]-0.3; AX2=AX1+AW+0.2; AX3=AX2+AW+0.2
AY=4.55
rbox(AX1,AY,AW,AH,"#EAFAF1","h->j  encodes\ndist( origin -> vj )",
     fontsize=8,border=GRN,lw=1.1)
rbox(AX2,AY,AW,AH,"#FEF9E7","h<-i  encodes\ndist( vi -> dest )",
     fontsize=8,border=ORG,lw=1.1)
rbox(AX3,AY,AW,AH,C_POOL,"Pool  d~\n= min over pool\n~= d*(vi,vj)",
     fontsize=8,border=RED,lw=1.1)
ax.text(AX3+AW/2,AY-0.22,"no road network",ha="center",
        fontsize=7.2,color=RED,style="italic")

darr(xs[5]+NW/2,6.33,AX1+AW/2,AY+AH)
darr(xs[2]+NW/2,5.53,AX2+AW/2,AY+AH)

rbox(AX1,3.25,AX3+AW-AX1,0.88,C_UBS,
     "UBS check:   ( d->j - d->i )  <=  (1+e) * d~\n"
     "BiGRU implicitly covers every subpath  --  no O(L^2) queries",
     fontsize=8.5,border=GOLD,lw=1.3)

arr(AX1+AW/2,AY,AX1+AW/2,4.13,color=GRN,lw=1.0,ls="dashed")
arr(AX2+AW/2,AY,AX2+AW/2,4.13,color=ORG,lw=1.0,ls="dashed")
arr(AX3+AW/2,AY,AX3+AW/2,4.13,color=RED,lw=1.0,ls="dashed")

ax.axvline(8.1,color="#CCCCCC",lw=1.2,linestyle="--",ymin=0.02,ymax=0.97)
ax.text(13.5,8.75,"Scoring Pipeline",ha="center",fontsize=11,fontweight="bold")

PW=3.2; PH=0.75; PXR=11.9
steps=[
    (7.65,C_NODE, "Candidate Paths\n(any generator)",  BORDER,False),
    (6.55,C_FWD,  "Fwd GRU  ->\n(origin-side  d,  d~)",GRN,  False),
    (5.65,C_BWD,  "<-  Bwd GRU\n(dest-side  d,  d~)",  ORG,  False),
    (4.55,"#F2F3F4","Concat  [ h->n | h<-1 ]",         BORDER,False),
    (3.45,C_MLP,  "MLP Scoring Head\nscore in [0,100]", BORDER,False),
    (2.35,C_DIV,  "Diversity Filter\n(overlap <= theta)",BORDER,False),
    (1.25,C_OUT,  "Top-k Diverse Routes",               BORDER,True),
]
for(y,c,lbl,bd,bo) in steps:
    rbox(PXR,y,PW,PH,c,lbl,fontsize=9,bold=bo,border=bd)

cx=PXR+PW/2
for i in range(len(steps)-1):
    arr(cx,steps[i][0],cx,steps[i+1][0]+PH,lw=1.5)

rbox(PXR+PW+0.15,5.75,1.9,0.95,C_POOL,
     "Pool  d~\n(from candidate\nset)",fontsize=8,border=RED,lw=1.2)
arr(PXR+PW+0.15+0.95,5.75+0.68,PXR+PW,steps[1][0]+PH/2,color=RED,lw=1.1,ls="dashed")
arr(PXR+PW+0.15+0.95,5.75+0.27,PXR+PW,steps[2][0]+PH/2,color=RED,lw=1.1,ls="dashed")

legend_items=[
    mpatches.Patch(facecolor=C_FWD, edgecolor=GRN,  label="Forward GRU (origin-side)"),
    mpatches.Patch(facecolor=C_BWD, edgecolor=ORG,  label="Backward GRU (dest-side)"),
    mpatches.Patch(facecolor=C_POOL,edgecolor=RED,  label="Pool distances  d~"),
    mpatches.Patch(facecolor=C_SPAN,edgecolor=RED,  label="Subpath (vi, vj)"),
    mpatches.Patch(facecolor=C_UBS, edgecolor=GOLD, label="UBS implicit check"),
]
ax.legend(handles=legend_items,loc="lower left",fontsize=8,
          framealpha=0.95,bbox_to_anchor=(0.0,0.0))

import os; os.makedirs("figures",exist_ok=True)
plt.tight_layout(pad=0.2)
plt.savefig("figures/arch_sample.png",dpi=160,bbox_inches="tight")
print("Saved -> figures/arch_sample.png")
