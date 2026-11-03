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
    # ---------- 加载图像并初始化报告 ----------
    image=OUT/'figure4_native.png'
    rgb=np.asarray(Image.open(image).convert('RGB')).astype(float)
    # 报告元数据: 明确声明这是"可见线段数字化 + 遮挡段重建", 非原始测量
    report={'label':'Figure 4 visible mean-line digitization + occluded-segment reconstruction; NOT original measurements',
            'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
            'normal_color':'blue','x':'normalized cycle %, peak hip extension as cycle origin',
            'angles':'sagittal relative joint angle, degrees, hip/knee flexion positive',
            'endpoint_rule':'periodic closure: endpoint value is mean of available start/end local observations; not a table-2 constraint',
            'joints':{}}
    # 相位网格 501 点(0~1 归一化步态周期), results 缓存每关节的曲线与可见性
    phase=np.linspace(0,1,501); results={}
    fig,axes=plt.subplots(2,2,figsize=(14,8))  # 2 行(髋/膝) x 2 列(原图叠加/角度曲线)

    # ---------- 逐关节数字化(髋 / 膝) ----------
    # 每关节元组: (名称, x轴像素范围, y轴角度上限, Table2 参考[min,max,ROM,peak_timing])
    for i,(name,x0,x1,ymax,table) in enumerate([
        ('hip',126,640,80,[-1.16,50.19,51.35,51.9]),
        ('knee',770,1280,100,[-.75,67.51,68.26,37.5])]):
        y0,y1=183,369  # 两关节共用的 y 像素范围

        # --- 提取蓝色均值线像素 ---
        # 裁剪目标区域并按 R/G/B 分离通道
        a=rgb[y0+1:y1-1,x0+2:x1-1];r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
        # 蓝色阈值: 蓝-红差>60, 蓝-绿差>20, 红<100, 蓝>90
        blue=(b-r>60)&(b-g>20)&(r<100)&(b>90)
        px=[];py=[]
        for x in range(blue.shape[1]):
            # 每列取所有蓝色像素的中位数 y, 作为该 x 处的均值线高度
            yy=np.flatnonzero(blue[:,x])
            if len(yy): px.append(x0+2+x);py.append(y0+1+float(np.median(yy)))

        # --- 像素坐标 -> 角度 ---
        px=np.array(px);py=np.array(py);xp=(px-x0)/(x1-x0)  # x 归一化到 [0,1]
        # y 像素映射到角度范围 [-20, ymax]; y 向下为正, 故用 (y1-py) 翻转
        angles=-20+(y1-py)/(y1-y0)*(ymax+20)

        # --- 周期性端点闭合 + 插值重建 ---
        # Preserve source observations separately; smooth pixel stair-steps only.
        # Interpolation across a gap is explicitly tracked, never relabelled visible.
        # 端点值 = 首尾局部观测均值, 用于周期闭合(非 Table2 约束)
        endpoint=float(np.mean(np.r_[angles[xp<.015],angles[xp>.985]]))
        xpfit=np.r_[0,xp,1]; qfit=np.r_[endpoint,angles,endpoint]  # 端点处补 0/1
        raw=PchipInterpolator(xpfit,qfit)(phase)                    # 主重建: PCHIP(保形单调)
        alternative=CubicSpline(xpfit,qfit,bc_type='periodic')(phase)  # 敏感性: 周期样条

        # --- Savitzky-Golay 平滑(周期边界) ---
        # Periodic 4%-cycle Savitzky-Golay smoothing removes raster stair-step noise.
        # 窗口 21 点(≈4% 周期), 3 阶多项式, wrap 模式保证周期连续
        smooth=savgol_filter(raw[:-1],21,3,mode='wrap'); smooth=np.r_[smooth,smooth[0]]
        alt=savgol_filter(alternative[:-1],21,3,mode='wrap');alt=np.r_[alt,alt[0]]

        # --- 标记可见段 vs 重建段(缺口) ---
        # phase 点距最近观测像素的距离 <= 1.1 像素宽度 才算"可见"
        visible=np.array([np.min(abs(xp-p)) <= 1.1/(x1-x0) for p in phase])
        # 可见段保留观测值(线性插值), 缺口置 NaN
        observed=np.interp(phase,xp,angles); observed[~visible]=np.nan
        # 扫描 visible 数组, 提取所有连续缺口区间(百分比)
        gaps=[]; st=None
        for k,good in enumerate(np.r_[visible,True]):
            if not good and st is None:st=k
            if good and st is not None:gaps.append([float(100*phase[st]),float(100*phase[min(k,500)])]);st=None

        results[name]=(smooth,alt,visible,observed)

        # --- 计算指标并写入报告 ---
        report['joints'][name]={'axis_pixels':[x0,x1,y0,y1],'y_range_deg':[-20,ymax],
            'visible_fraction':float(visible.mean()),'reconstruction_intervals_percent':gaps,
            'min_deg':float(smooth.min()),'max_deg':float(smooth.max()),'ROM_deg':float(np.ptp(smooth)),
            'peak_flexion_cycle_percent':float(100*phase[np.argmax(smooth)]),
            'table2_min_max_ROM_peak_timing':table,
            # 与 Table2 的偏差: [min, max, ROM, peak_timing]
            'table2_difference_min_max_ROM_peak_timing':(np.array([smooth.min(),smooth.max(),np.ptp(smooth),100*phase[np.argmax(smooth)]])-table).tolist(),
            'pchip_vs_cubic_max_difference_deg':float(np.max(abs(smooth-alt))),  # 重建方法敏感性
            'visible_fit_rmse_deg':float(np.sqrt(np.mean((smooth[visible]-observed[visible])**2)))}  # 可见段拟合误差

        # --- 派生角速度(对角度求导 × 平均频率 0.67 Hz) ---
        velocity=CubicSpline(phase,smooth,bc_type='periodic')(phase,1)*.67
        table_velocity=[98.85,-131.50] if name=='hip' else [140.52,-184.48]
        report['joints'][name]['derived_velocity_peak_max_min_deg_s']=[float(velocity.max()),float(velocity.min())]
        report['joints'][name]['table2_velocity_peak_max_min_deg_s']=table_velocity
        # 明确声明: 这是重建角度的导数, 非 Figure4 d/e 数字化, 不强制匹配 Table2
        report['joints'][name]['velocity_note']='Derivative of reconstructed angle at mean frequency; NOT digitized Figure4 d/e, not forced to match table2'

        # --- 绘图: 左列原图叠加, 右列角度曲线 ---
        ax=axes[i,0]; ax.imshow(rgb.astype(np.uint8));ax.scatter(px,py,s=2,c='lime',label='Visible blue pixels')
        ax.plot(x0+phase*(x1-x0),y1-(smooth+20)/(ymax+20)*(y1-y0),'m--',lw=1,label='Reconstructed + smoothed')
        ax.set_xlim(x0-10,x1+10);ax.set_ylim(y1+10,y0-10);ax.set_title(name+' source overlay');ax.legend(fontsize=7)
        ax=axes[i,1];ax.plot(phase*100,smooth,label='PCHIP + smoothing');ax.plot(phase*100,alt,'--',label='Cubic sensitivity')
        ax.scatter(phase[visible]*100,observed[visible],s=2,label='Visible observations')
        for l,h in gaps:ax.axvspan(l,h,color='orange',alpha=.2)  # 橙色 = 重建缺口
        ax.set(xlabel='Cycle (%)',ylabel='Relative angle (deg)',title=name+'; orange = reconstructed');ax.legend(fontsize=7);ax.grid(alpha=.2)

    # ---------- 写出 CSV / JSON 报告 / 校验图 ----------
    with (OUT/'yang_normal_reconstructed.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['cycle_fraction','hip_deg','knee_deg','hip_alt_deg','knee_alt_deg','hip_visible','knee_visible','hip_observed_deg','knee_observed_deg'])
        for k,p in enumerate(phase):w.writerow([p,results['hip'][0][k],results['knee'][0][k],results['hip'][1][k],results['knee'][1][k],int(results['hip'][2][k]),int(results['knee'][2][k]),results['hip'][3][k],results['knee'][3][k]])
    (OUT/'digitization_report.json').write_text(json.dumps(report,indent=2))
    fig.tight_layout();fig.savefig(OUT/'digitization_check.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()