"""Visible blue mean-line extraction; explicitly labelled gap reconstruction."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'paper_reference'

def main():
    # ---------- 载入源图与元数据 ----------
    image=OUT/'figure4_native.png'
    rgb=np.asarray(Image.open(image).convert('RGB')).astype(float)
    # 报告头部: 明确标注这是"可见蓝线数字化 + 遮挡段重建", 并非原始测量数据
    # 记录图像 SHA-256 以便追溯, 并写明坐标/单位/端点闭合约定
    report={'label':'Figure 4 visible mean-line digitization + occluded-segment reconstruction; NOT original measurements',
            'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
            'normal_color':'blue','x':'normalized cycle %, peak hip extension as cycle origin',
            'angles':'sagittal relative joint angle, degrees, hip/knee flexion positive',
            'endpoint_rule':'periodic closure: endpoint value is mean of available start/end local observations; not a table-2 constraint',
            'joints':{}}
    phase=np.linspace(0,1,501); results={}  # 归一化到 501 点的周期相位网格
    fig,axes=plt.subplots(2,2,figsize=(14,8))  # 每行一个关节: 左=原图叠加, 右=曲线对比

    # 两个关节的取图参数: 关节名, 横轴像素范围[x0,x1], 纵轴角度上限 ymax,
    # 以及论文 Table 2 的参考值[min,max,ROM,峰值时刻%](仅作对比, 不作为约束)
    for i,(name,x0,x1,ymax,table) in enumerate([
        ('hip',126,640,80,[-1.16,50.19,51.35,51.9]),
        ('knee',770,1280,100,[-.75,67.51,68.26,37.5]])):
        y0,y1=183,369  # 纵轴像素范围(所有子图共享)

        # ---------- 提取蓝色均值线 ----------
        a=rgb[y0+1:y1-1,x0+2:x1-1];r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
        # 蓝色像素判定: 蓝通道显著高于红/绿, 且红较低(排除浅色背景与黑色文字)
        blue=(b-r>60)&(b-g>20)&(r<100)&(b>90)
        px=[];py=[]
        for x in range(blue.shape[1]):
            # 逐列取所有蓝色像素的中位行号作为该列均值线位置
            yy=np.flatnonzero(blue[:,x])
            if len(yy): px.append(x0+2+x);py.append(y0+1+float(np.median(yy)))
        px=np.array(px);py=np.array(py);xp=(px-x0)/(x1-x0)  # 横向归一化为周期比例
        # 像素纵坐标 -> 角度: 纵轴范围[-20, ymax], 图像 y 向下增大故需翻转
        angles=-20+(y1-py)/(y1-y0)*(ymax+20)

        # Preserve source observations separately; smooth pixel stair-steps only.
        # Interpolation across a gap is explicitly tracked, never relabelled visible.
        # 端点闭合: 用首尾各 1.5% 周期内可用观测的均值, 而非 Table 2 数值
        endpoint=float(np.mean(np.r_[angles[xp<.015],angles[xp>.985]]))
        xpfit=np.r_[0,xp,1]; qfit=np.r_[endpoint,angles,endpoint]  # 在 0 和 1 处补端点值形成闭环
        # 主方案: PCHIP 单调保形插值, 避免在稀疏/遮挡区间产生过冲
        raw=PchipInterpolator(xpfit,qfit)(phase)
        # 敏感性对照: 周期三次样条, 用于评估插值方式对结果的影响
        alternative=CubicSpline(xpfit,qfit,bc_type='periodic')(phase)

        # Periodic 4%-cycle Savitzky-Golay smoothing removes raster stair-step noise.
        # 周期边界下做 21 点(约4%周期)、3 阶 SG 平滑, 仅去除像素栅格阶梯噪声;
        # 先对前 500 点平滑再复制首点到末尾, 保证严格周期闭合
        smooth=savgol_filter(raw[:-1],21,3,mode='wrap'); smooth=np.r_[smooth,smooth[0]]
        alt=savgol_filter(alternative[:-1],21,3,mode='wrap');alt=np.r_[alt,alt[0]]

        # ---------- 标记可见/遮挡区间 ----------
        # 若某相位点与最近真实观测的横向距离在约 1.1 像素内, 视为"可见"
        visible=np.array([np.min(abs(xp-p)) <= 1.1/(x1-x0) for p in phase])
        # 可见点保留真实观测值, 其余置 NaN(遮挡段不冒充观测)
        observed=np.interp(phase,xp,angles); observed[~visible]=np.nan
        # 扫描 visible 布尔序列, 收集所有遮挡区间(以周期百分比表示)
        gaps=[]; st=None
        for k,good in enumerate(np.r_[visible,True]):
            if not good and st is None:st=k
            if good and st is not None:gaps.append([float(100*phase[st]),float(100*phase[min(k,500)])]);st=None
        results[name]=(smooth,alt,visible,observed)

        # ---------- 汇总指标与 Table 2 对比 ----------
        report['joints'][name]={'axis_pixels':[x0,x1,y0,y1],'y_range_deg':[-20,ymax],
            'visible_fraction':float(visible.mean()),'reconstruction_intervals_percent':gaps,
            'min_deg':float(smooth.min()),'max_deg':float(smooth.max()),'ROM_deg':float(np.ptp(smooth)),
            'peak_flexion_cycle_percent':float(100*phase[np.argmax(smooth)]),
            'table2_min_max_ROM_peak_timing':table,
            # 重建值减去 Table 2 参考值的差(用于透明披露偏差)
            'table2_difference_min_max_ROM_peak_timing':(np.array([smooth.min(),smooth.max(),np.ptp(smooth),100*phase[np.argmax(smooth)]])-table).tolist(),
            # PCHIP 与周期三次样条的最大偏差(插值方式敏感性)
            'pchip_vs_cubic_max_difference_deg':float(np.max(abs(smooth-alt))),
            # 平滑曲线在可见区间上相对真实观测的拟合 RMSE
            'visible_fit_rmse_deg':float(np.sqrt(np.mean((smooth[visible]-observed[visible])**2)))}

        # ---------- 由重建角度求角速度 ----------
        # 对周期样条求一阶导并乘以平均频率 .67 Hz, 得到 deg/s
        velocity=CubicSpline(phase,smooth,bc_type='periodic')(phase,1)*.67
        table_velocity=[98.85,-131.50] if name=='hip' else [140.52,-184.48]
        report['joints'][name]['derived_velocity_peak_max_min_deg_s']=[float(velocity.max()),float(velocity.min())]
        report['joints'][name]['table2_velocity_peak_max_min_deg_s']=table_velocity
        # 明确说明: 角速度来自重建曲线的求导, 未对 Table 2 强行对齐
        report['joints'][name]['velocity_note']='Derivative of reconstructed angle at mean frequency; NOT digitized Figure4 d/e, not forced to match table2'

        # ---------- 绘图: 左=原图叠加, 右=曲线对比 ----------
        ax=axes[i,0]; ax.imshow(rgb.astype(np.uint8));ax.scatter(px,py,s=2,c='lime',label='Visible blue pixels')
        # 将重建角度反投影回像素坐标, 叠加到原图上核对对齐
        ax.plot(x0+phase*(x1-x0),y1-(smooth+20)/(ymax+20)*(y1-y0),'m--',lw=1,label='Reconstructed + smoothed')
        ax.set_xlim(x0-10,x1+10);ax.set_ylim(y1+10,y0-10);ax.set_title(name+' source overlay');ax.legend(fontsize=7)
        ax=axes[i,1];ax.plot(phase*100,smooth,label='PCHIP + smoothing');ax.plot(phase*100,alt,'--',label='Cubic sensitivity')
        ax.scatter(phase[visible]*100,observed[visible],s=2,label='Visible observations')
        # 用橙色带标出重建(遮挡)区间, 与真实观测视觉区分
        for l,h in gaps:ax.axvspan(l,h,color='orange',alpha=.2)
        ax.set(xlabel='Cycle (%)',ylabel='Relative angle (deg)',title=name+'; orange = reconstructed');ax.legend(fontsize=7);ax.grid(alpha=.2)

    # ---------- 导出重建曲线 CSV ----------
    # 同时写出主方案/敏感性方案, 并附可见标志与原始观测(NaN 表示遮挡)
    with (OUT/'yang_normal_reconstructed.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['cycle_fraction','hip_deg','knee_deg','hip_alt_deg','knee_alt_deg','hip_visible','knee_visible','hip_observed_deg','knee_observed_deg'])
        for k,p in enumerate(phase):w.writerow([p,results['hip'][0][k],results['knee'][0][k],results['hip'][1][k],results['knee'][1][k],int(results['hip'][2][k]),int(results['knee'][2][k]),results['hip'][3][k],results['knee'][3][k]])

    # ---------- 写出报告与检查图 ----------
    (OUT/'digitization_report.json').write_text(json.dumps(report,indent=2))
    fig.tight_layout();fig.savefig(OUT/'digitization_check.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()