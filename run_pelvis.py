"""Run verified-paper approximate reference and hypothetical pitch experiments."""
import argparse, json, itertools
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assistance_sim import ROOT, MODES, safe_output
from assistance_analysis import compare
from pelvis_runner import run_pelvis

def main():
    # ---------- 命令行参数定义 ----------
    p=argparse.ArgumentParser()
    # 实验套件: pilot(小规模探索)/full(完整网格)/single(单组参数)
    p.add_argument('--suite',choices=['pilot','full','single'],default='pilot')
    # 运行模式: 各辅助模式或全部; 幅值(度), 频率比, 相位(度)
    p.add_argument('--mode',choices=[*MODES,'all'],default='all');p.add_argument('--amplitude',type=float,default=2)
    p.add_argument('--ratio',type=float,default=1);p.add_argument('--phase',type=float,default=0)
    # 仿真持续时间(秒); 是否开启交互查看器
    p.add_argument('--duration',type=float,default=24);p.add_argument('--viewer',action='store_true')
    # 参考轨迹来源: paper(论文重建)/alternate(备选)/legacy(旧版)
    p.add_argument('--reference',choices=['paper','alternate','legacy'],default='paper')
    # 输出目录
    p.add_argument('--out',type=Path,default=ROOT/'pelvis_output');args=p.parse_args()

    # ---------- 参数合法性校验 ----------
    # 查看器只能在单套件+单模式下使用
    if args.viewer and (args.suite!='single' or args.mode=='all'):p.error('Viewer requires single suite and one mode')

    # ---------- 加载配置 ----------
    # 读取主配置与辅助配置, 并用命令行持续时间覆盖
    c=json.loads((ROOT/'config.json').read_text());ac=json.loads((ROOT/'assistance_config.json').read_text());c['duration_s']=args.duration
    # 创建安全输出目录(避免覆盖已有结果)
    out=safe_output(args.out)
    # 保护 assistance_output 目录不被误覆盖
    if out==ROOT/'assistance_output' or ROOT/'assistance_output' in out.parents:raise ValueError('Preserve assistance output')

    # ---------- 构建实验用例 (幅值, 频率比, 相位) ----------
    cases=[(0,1,0),(2,1,0),(5,.5,90),(5,2,180)]  # pilot 默认 4 组
    if args.suite=='full':cases=[(0,1,0)]+list(itertools.product([2,5],[.5,1,2],[0,90,180,270]))  # full: 1 + 24 组网格
    if args.suite=='single':cases=[(args.amplitude,args.ratio,args.phase)]  # single: 仅命令行指定的一组

    # ---------- 逐个用例运行仿真 ----------
    all_metrics={}
    for amp,ratio,phase in cases:
        key=f'A{amp:g}_r{ratio:g}_p{phase:g}'  # 用例标识: 幅值_频率比_相位
        # 构造假设性骨盆俯仰扰动(非 Yang 实测, 仅用于灵敏度分析)
        disturbance={'amplitude_deg':amp,'frequency_hz':ratio*c['frequency_hz'],'phase_deg':phase,'source':'hypothetical sensitivity test, not Yang measurement'}
        groups={}
        for mode in MODES if args.mode=='all' else [args.mode]:
            # 运行单个模式的骨盆仿真, 返回指标字典; full 套件不生成逐例绘图
            groups[mode]=run_pelvis(c,ac,mode,out/key/mode,disturbance,args.reference,viewer=args.viewer,plots=args.suite!='full')
            # 实时打印四关节跟踪 RMSE
            print(key,mode,'tracking',groups[mode]['aggregate']['tracking_rmse_deg'],flush=True)
        all_metrics[key]=groups
        # 多模式下对同一用例的不同模式结果做对比分析
        if args.mode=='all':compare(out/key)

    # ---------- 保存汇总指标 ----------
    (out/'suite_metrics.json').write_text(json.dumps(all_metrics,indent=2))

    # ---------- 绘制跨用例对比图(1行3列) ----------
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for mode in (MODES if args.mode=='all' else [args.mode]):
        keys=list(all_metrics)
        # 三个子图分别: 左髋 IMU 感知 RMSE / 四关节跟踪 RMSE / 基座净功
        for ax,values in zip(axes,[[all_metrics[k][mode]['joints']['L_hip']['imu_rmse_deg'] for k in keys],
                                  [all_metrics[k][mode]['aggregate']['tracking_rmse_deg'] for k in keys],
                                  [all_metrics[k][mode]['base']['net_work_J'] for k in keys]]):
            ax.plot(keys,values,'o-',label=mode);ax.tick_params(axis='x',rotation=35)
    for ax,title in zip(axes,['Left hip sensing RMSE (deg)','Four-joint tracking RMSE (deg)','Prescribed base net work (J)']):ax.set_title(title);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Hypothetical pelvis pitch; reconstructed paper reference');fig.tight_layout();fig.savefig(out/'pelvis_comparison.png',dpi=150)

if __name__=='__main__':main()