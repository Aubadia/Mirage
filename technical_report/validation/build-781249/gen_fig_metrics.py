"""Reproduce quantitative figures from source-linked metrics; run through Slurm."""
import os,pathlib,json
ROOT=pathlib.Path(__file__).resolve().parents[1]
if not os.environ.get('SLURM_JOB_ID'): raise SystemExit('Run plotting inside a Slurm allocation.')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
D=json.loads((ROOT/'data/metrics.json').read_text())['records']
V={(r['phase'],r['arm'],r['metric']):r['value'] for r in D}
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300,'axes.axisbelow':True})
COLORS=['#0072B2','#D55E00']; ARMS=['reference','candidate'];HATCH=['','///']
def save(fig,name):
 fig.savefig(ROOT/'figures'/('fig_'+name+'.pdf'),bbox_inches='tight')
 target=ROOT/'figures'/('fig_'+name+'.png');fig.savefig(target,dpi=300,bbox_inches='tight')
 with Image.open(target) as im:im.convert('L').save(ROOT/'validation'/('fig_'+name+'-grayscale.png'))
 plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(7.05,2.35))
for ax,p in zip(axs,['008','009','010']):
 x=np.arange(3)
 for i,a in enumerate(ARMS):
  vals=[V[p,a,k] for k in ['attempts','valid','mr_applicable']]
  bars=ax.bar(x+(i-.5)*.34,vals,.31,color=COLORS[i],hatch=HATCH[i],label=a.capitalize(),edgecolor='white',linewidth=.5)
  for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+max(V[p,a2,'attempts'] for a2 in ARMS)*(.025 if i==0 else .15),str(v),ha='center',fontsize=7)
 ax.set_xticks(x,['Attempts','Valid\nsource','Applicable\nMR']);ax.set_ylabel('Cases / attempts');ax.set_ylim(0,max(V[p,a,'attempts'] for a in ARMS)*1.24)
 ax.text(0,1.04,'Roadmap '+p,transform=ax.transAxes,va='bottom',fontsize=9);ax.grid(axis='y',alpha=.15)
axs[1].legend(loc='upper center',bbox_to_anchor=(.5,1.53),ncol=2)
fig.subplots_adjust(wspace=.5,bottom=.22,top=.70);save(fig,'case_funnels')
fig,axs=plt.subplots(1,3,figsize=(7.05,2.45))
spec=[('008',['unary_units','joint_units'],['Unary','Joint']),('009',['qualified_joint','planned_valid','memory_goal_attained'],['Q-joint','Planned','Memory']),('010',['semantic_units','chain_units','gradient_update_attained'],['Operand\nunits','Chain\nunits','Goal\ncases'])]
for ax,(p,keys,labels) in zip(axs,spec):
 x=np.arange(len(keys));mx=max(V[p,a,k] for a in ARMS for k in keys)
 for i,a in enumerate(ARMS):
  vals=[V[p,a,k] for k in keys];bars=ax.bar(x+(i-.5)*.34,vals,.31,color=COLORS[i],hatch=HATCH[i],edgecolor='white',linewidth=.5,label=a.capitalize())
  for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+mx*(.025 if i==0 else .15),str(v),ha='center',fontsize=7)
 ax.set_xticks(x,labels)
 if p=='009':ax.set_xticklabels(labels,rotation=25,ha='right',rotation_mode='anchor')
 ax.set_ylim(0,mx*1.30);ax.set_ylabel('Observed count');ax.text(0,1.04,'Roadmap '+p,transform=ax.transAxes,va='bottom');ax.grid(axis='y',alpha=.15)
axs[1].legend(loc='upper center',bbox_to_anchor=(.5,1.53),ncol=2)
fig.subplots_adjust(wspace=.53,bottom=.22,top=.70);save(fig,'coverage_tradeoffs')
fig,ax=plt.subplots(figsize=(7.05,2.65))
status=['agreement','violation','unsupported','unknown']; palette=['#009E73','#D55E00','#E69F00','#9b9b9b'];hatches=['','xx','///','...']; y=0;labels=[]
for p in ['008','009','010']:
 for a in ARMS:
  vals=[V[p,a,'audit_'+k] for k in status];total=sum(vals);left=0
  for k,(v,c,h) in enumerate(zip(vals,palette,hatches)):
   width=100*v/total
   ax.barh(y,width,left=left,color=c,hatch=h,height=.64,edgecolor='white',linewidth=.5,label=status[k].capitalize() if y==0 else None)
   if width>=6:ax.text(left+width/2,y,str(v),ha='center',va='center',fontsize=8)
   left+=width
  ax.text(101,y,'n='+str(total),va='center',fontsize=8);labels.append(p+' / '+a);y+=1
ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(0,112);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Share of saved numerical-audit member status records (%)');ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.22));ax.spines['bottom'].set_visible(False)
fig.subplots_adjust(left=.24,right=.97,top=.8,bottom=.23);save(fig,'audit_statuses')
print('Generated three figures as PDF/300-DPI PNG, with grayscale review copies.')
