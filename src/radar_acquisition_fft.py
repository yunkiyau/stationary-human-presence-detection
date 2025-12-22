# - Keybinds: [D]=DC toggle, [W]=complex window toggle, [R]=record, [Q]=quit

import serial, struct, time, csv, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --- PORT / BAUD ---
PORT = 'COM6'      # Change to local port
BAUD = 230400

# --- RAW-ADC configuration ---
S = '!S150B2812'
F = '!F000168F0'
P = '!P00000603'
B = '!B2452C122'

# === FFT SETTINGS ===
NFFT                = 512
INCLUDE_NYQ         = True
VIEW_MODE           = "magnitude"
APPLY_WINDOW_IQ     = True
APPLY_WINDOW_CPLX   = True

# --- RANGE / GATE SETTINGS ---
C_LIGHT = 299_792_458.0
FS_ADC  = 675000.0
B_RAMP  = 3.078e9
T_RAMP  = 0.000849
SLOPE   = B_RAMP / T_RAMP
R_GATE  = (1.3, 1.7)

def bin_to_range_per_bin(nfft=NFFT, fs=FS_ADC, slope=SLOPE):
    return (C_LIGHT * fs) / (2.0 * slope * nfft)

def k_to_m(k, nfft=NFFT, fs=FS_ADC, slope=SLOPE):
    fb = (np.asarray(k) * fs) / nfft
    return (C_LIGHT * fb) / (2.0 * slope)

def m_to_k(m, nfft=NFFT, fs=FS_ADC, slope=SLOPE):
    fb = (2.0 * slope * np.asarray(m)) / C_LIGHT
    return (fb * nfft) / fs

# --- Phase-tracking config ---
PHASE_WINDOW_SEC = 30.0
PHASE_ALPHA      = 1.0
USE_MEDIAN_MASK  = True

# --- Protocol helpers ---
HDR = b'\xAA\xAA\xBB\xCC'
def U16(b): return struct.unpack('<H', b)[0]
def U32(b): return struct.unpack('<I', b)[0]

def send(sp, cmd: str):
    sp.write((cmd + '\r\n').encode('ascii')); sp.flush()

def parse_one_frame(buf: bytes):
    idx = buf.find(HDR)
    if idx < 0: return None, (buf[-3:] if len(buf) > 3 else buf)
    if idx > 0: buf = buf[idx:]
    if len(buf) < 4 + 16: return None, buf
    off = 4
    frame_cnt    = U16(buf[off:off+2]); off += 2
    frame_id_u16 = U16(buf[off:off+2]); off += 2
    frame_len    = U16(buf[off:off+2]); off += 2
    txid         = buf[off]; off += 1
    rxid         = buf[off]; off += 1
    datasrc      = buf[off]; off += 1
    gain_db      = buf[off]; off += 1
    meas_cnt     = U16(buf[off:off+2]); off += 2
    slow_cnt     = U16(buf[off:off+2]); off += 2
    upd_rate     = U16(buf[off:off+2]); off += 2
    need_total = off + frame_len + 6
    if len(buf) < need_total: return None, buf
    data_type = buf[off]; off += 1
    var_type  = buf[off]; off += 1
    no_elem   = U16(buf[off:off+2]); off += 2
    payload   = buf[off:off + (frame_len - 4)]; off += (frame_len - 4)
    crc32     = U32(buf[off:off+4]); off += 4
    stop      = buf[off:off+2];       off += 2
    fid = 'D' if b'D' in struct.pack('<H', frame_id_u16) else f'0x{frame_id_u16:04X}'
    frame = {'fid': fid, 'cnt': frame_cnt, 'ds': datasrc,
             'payload': payload, 'meas_cnt': meas_cnt, 'slow_cnt': slow_cnt,
             'txid': txid, 'rxid': rxid, 'gain_db': gain_db, 'upd_rate': upd_rate}
    return frame, buf[off:]

def to_i16(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype='<i2').astype(np.int32)

# --- FFT helpers ---
def blackman_harris_4term(N: int) -> np.ndarray:
    a0,a1,a2,a3 = 0.35875,0.48829,0.14128,0.01168
    n = np.arange(N)
    return (a0 - a1*np.cos(2*np.pi*n/(N-1))
              + a2*np.cos(4*np.pi*n/(N-1))
              - a3*np.cos(6*np.pi*n/(N-1)))

def pos_half_bins_from(X: np.ndarray, include_nyq: bool=True):
    N = X.size; half = N//2
    return (np.arange(half+1), X[:half+1]) if include_nyq else (np.arange(half), X[:half])

def apply_view_mode(Xh: np.ndarray, mode: str) -> np.ndarray:
    return np.abs(Xh) if mode == "magnitude" else np.real(Xh)

def fft_last_block_real(x: np.ndarray, nfft: int, dc_cancel: bool, use_bh: bool) -> np.ndarray:
    if x.size < 1: return np.array([], dtype=np.complex128)
    xb = (x[-nfft:] if x.size >= nfft else np.pad(x.astype(np.float64),(0,nfft-x.size))).astype(np.float64)
    if dc_cancel: xb = xb - np.mean(xb)
    if use_bh: xb = xb * blackman_harris_4term(nfft)
    return np.fft.fft(xb, n=nfft)

def fft_last_block_complex(xc: np.ndarray, nfft: int, dc_cancel: bool, use_bh: bool) -> np.ndarray:
    if xc.size < 1: return np.array([], dtype=np.complex128)
    xb = (xc[-nfft:] if xc.size >= nfft else np.pad(xc.astype(np.complex128),(0,nfft-xc.size)))
    if dc_cancel: xb = xb - np.mean(xb)
    if use_bh: xb = xb * blackman_harris_4term(nfft)
    return np.fft.fft(xb, n=nfft)

def main():
    cfg = {'dc_remove': True, 'w_cplx': APPLY_WINDOW_CPLX}
    rec_active=False; record_file=None; record_writer=None; record_path=None; rec_start_time=None
    sp=serial.Serial(PORT, baudrate=BAUD, bytesize=8, parity='N', stopbits=1, timeout=0.05)
    time.sleep(0.2); sp.reset_input_buffer(); sp.reset_output_buffer()
    for cmd in (S,F,P,B): send(sp,cmd); time.sleep(0.03)
    print("✅ Config sent. [D]=DC, [W]=complex window, [R]=record, [Q]=quit")

    plt.style.use('dark_background')
    fig=plt.figure(figsize=(12,8),constrained_layout=True)
    gs=GridSpec(2,2,figure=fig)
    ax_td=fig.add_subplot(gs[0,0]); ax_fft=fig.add_subplot(gs[1,0])
    ax_c=fig.add_subplot(gs[0,1]); ax_res=fig.add_subplot(gs[1,1])

    # ADC time-domain
    line_I,=ax_td.plot([],[],'#FF5E00',lw=1.0,label='I (ADC)')
    line_Q,=ax_td.plot([],[],'#00D1FF',lw=1.0,label='Q (ADC)')
    ax_td.set_title("ADC I & Q (live)")
    ax_td.set_xlabel("Sample index"); ax_td.set_ylabel("ADC counts")
    ax_td.grid(True,ls=':',lw=0.6); ax_td.legend(loc='lower right')
    ax_td.set_ylim(-10000,10000)

    # Gate bins
    dR=bin_to_range_per_bin()
    k_lo_f = m_to_k(R_GATE[0])                     
    k_hi_f = m_to_k(R_GATE[1])                     
    k_lo_f = max(0.0, min(k_lo_f, NFFT//2))        
    k_hi_f = max(k_lo_f + 1e-9, min(k_hi_f, NFFT//2))  

    # FFT(I)&(Q)
    ax_fft.set_title("FFT of I and Q Channels")
    line_fft_I,=ax_fft.plot([],[],'#FF5E00',lw=1.0,label='FFT(I) [BH]')
    line_fft_Q,=ax_fft.plot([],[],'#00D1FF',lw=1.0,label='FFT(Q) [BH]')
    ax_fft.set_xlabel(f"FFT bin"); ax_fft.set_ylabel("FFT value")
    ax_fft.grid(True,ls=':',lw=0.6); ax_fft.legend(loc='upper right'); ax_fft.set_xlim(0,NFFT//2)
    secax_iq=ax_fft.secondary_xaxis('top',functions=(k_to_m,m_to_k)); secax_iq.set_xlabel("Range (m)")
    ax_fft.axvspan(k_lo_f, k_hi_f, facecolor="#39FF14", alpha=0.25)      

    # Complex FFT
    ax_c.set_title("Complex Spectrum |FFT(I + jQ)|")
    line_fft_C,=ax_c.plot([],[],'#1E90FF',lw=1.1,label='|FFT(I+jQ)|')
    ax_c.set_xlabel(f"FFT bin"); ax_c.set_ylabel("FFT magnitude")
    ax_c.grid(True,ls=':',lw=0.6); ax_c.legend(loc='upper right'); ax_c.set_xlim(0,NFFT//2)
    secax=ax_c.secondary_xaxis('top',functions=(k_to_m,m_to_k)); secax.set_xlabel("Range (m)")
    ax_c.axvspan(k_lo_f, k_hi_f, facecolor="#39FF14", alpha=0.25)        

    # Phase plot
    ax_res.set_title("Unwrapped Phase From Range-Gated Z(t)")
    ax_res.set_xlabel("Time (s)"); ax_res.set_ylabel("Phase (rad)")
    ax_res.grid(True,ls=":",lw=0.6,alpha=0.3)
    line_phi,=ax_res.plot([],[],lw=1.8,color="#00FFF7")
    rec_text=ax_res.text(0.98,0.04,"",transform=ax_res.transAxes,ha='right',va='bottom',
                         color="#FF2DAA",fontsize=14,fontweight='bold',
                         bbox=dict(facecolor='black',alpha=0.35,boxstyle='round,pad=0.25'))
    timer_text=ax_res.text(0.98,0.11,"",transform=ax_res.transAxes,ha='right',va='bottom',
                           color="#FF2DAA",fontsize=14,fontweight='bold')
    #ax_res.set_ylim(-30,+30)

    t0=time.time(); t_hist=[]; phi_hist_unw=[]
    _last_phi_w=None; _last_phi_unw=None
    I_buf=np.array([],dtype=np.int32); Q_buf=np.array([],dtype=np.int32); MAX_BUF=8192

    def on_key(event):
        nonlocal rec_active,record_file,record_writer,record_path,rec_start_time
        if not event.key: return
        if event.key.lower()=='r':
            if not rec_active:
                ts=time.strftime("%Y%m%d_%H%M%S")
                record_path=os.path.abspath(f"phase_rec_{ts}.csv")
                record_file=open(record_path,"w",newline=""); record_writer=csv.writer(record_file)
                # Range axis for header
                bins=np.arange(NFFT//2+1); ranges=[k_to_m(k) for k in bins]
                header=["time_s","phi_wrapped_rad","phi_unwrapped_rad","Z_mag"]+[f"fft_bin_{i}@{ranges[i]:.3f}m" for i in bins]
                record_writer.writerow(header)
                rec_active=True; rec_start_time=time.time()
                rec_text.set_text("● REC"); timer_text.set_text("00:00")
                print(f"⏺ Recording to {record_path}")
            else:
                rec_active=False; rec_text.set_text(""); timer_text.set_text("")
                if record_file: record_file.flush(); record_file.close(); print(f"⏹ Saved {record_path}")
                record_file=None; record_writer=None; record_path=None; rec_start_time=None
        elif event.key.lower()=='q': plt.close(fig)

    fig.canvas.mpl_connect('key_press_event',on_key)

    buf=b''
    try:
        plt.show(block=False)
        while plt.fignum_exists(fig.number):
            chunk=sp.read(4096)
            if chunk: buf+=chunk
            updated=False
            while True:
                frame,buf=parse_one_frame(buf)
                if frame is None: break
                if frame['fid']!='D': continue
                arr=to_i16(frame['payload']); ds=frame['ds']
                if ds==3 and arr.size>=2:
                    n=(arr.size//2)*2; iq=arr[:n].reshape(-1,2)
                    I_new,Q_new=iq[:,0],iq[:,1]
                    I_buf=np.concatenate((I_buf,I_new))[-MAX_BUF:]
                    Q_buf=np.concatenate((Q_buf,Q_new))[-MAX_BUF:]
                    line_I.set_data(np.arange(I_new.size),I_new); line_Q.set_data(np.arange(Q_new.size),Q_new)
                    ax_td.set_xlim(0,max(I_new.size,Q_new.size)-1 if (I_new.size or Q_new.size) else 1)
                    updated=True
                elif ds==1:
                    I_buf=np.concatenate((I_buf,arr))[-MAX_BUF:]; line_I.set_data(np.arange(arr.size),arr)
                    if arr.size>1: ax_td.set_xlim(0,arr.size-1); updated=True
                elif ds==2:
                    Q_buf=np.concatenate((Q_buf,arr))[-MAX_BUF:]; line_Q.set_data(np.arange(arr.size),arr)
                    if arr.size>1: ax_td.set_xlim(0,arr.size-1); updated=True
            if updated and I_buf.size>=NFFT and Q_buf.size>=NFFT:
                # I/Q FFT
                XI=fft_last_block_real(I_buf,NFFT,dc_cancel=cfg['dc_remove'],use_bh=APPLY_WINDOW_IQ)
                XQ=fft_last_block_real(Q_buf,NFFT,dc_cancel=cfg['dc_remove'],use_bh=APPLY_WINDOW_IQ)
                binsI,XI_pos=pos_half_bins_from(XI,include_nyq=INCLUDE_NYQ)
                binsQ,XQ_pos=pos_half_bins_from(XQ,include_nyq=INCLUDE_NYQ)
                yI=apply_view_mode(XI_pos,VIEW_MODE); yQ=apply_view_mode(XQ_pos,VIEW_MODE)
                line_fft_I.set_data(binsI,yI); line_fft_Q.set_data(binsQ,yQ)
                # auto scale
                peak_iq=max(np.nanmax(yI) if yI.size else 0.0, np.nanmax(yQ) if yQ.size else 0.0)
                ax_fft.set_ylim(0,max(1.0,peak_iq*1.05))

                # Complex FFT
                L=min(I_buf.size,Q_buf.size)
                x_complex=I_buf[-L:].astype(np.float64)+1j*Q_buf[-L:].astype(np.float64)
                XC=fft_last_block_complex(x_complex,NFFT,dc_cancel=cfg['dc_remove'],use_bh=cfg['w_cplx'])
                binsC,XC_pos=pos_half_bins_from(XC,include_nyq=INCLUDE_NYQ)
                yC=np.abs(XC_pos); line_fft_C.set_data(binsC,yC)
                ax_c.set_ylim(0,max(1.0,np.nanmax(yC) if yC.size else 1.0))

                # before the gate-sum section (where you computed k_lo_f, k_hi_f already):
                k0 = max(0, int(np.floor(k_lo_f)))
                k1 = min(NFFT//2, int(np.ceil(k_hi_f)))

                # Gate-sum + phase unwrap
                if yC.size > k1 and k1 > k0:
                    mags = yC[k0:k1+1]
                    Xseg = XC_pos[k0:k1+1]
                    if USE_MEDIAN_MASK and mags.size:
                        mask = mags >= np.median(mags); mags = mags[mask]; Xseg = Xseg[mask]
                    if mags.size:
                        w = mags**(PHASE_ALPHA); w /= w.sum(); Z = (w*Xseg).sum()
                        phi = float(np.angle(Z)); magZ = float(np.abs(Z))
                        t_now = time.time() - t0
                        if _last_phi_w is None:
                            phi_unw = phi
                        else:
                            d = (phi - _last_phi_w + np.pi) % (2*np.pi) - np.pi
                            phi_unw = _last_phi_unw + d
                        _last_phi_w, _last_phi_unw = phi, phi_unw
                        t_hist.append(t_now); phi_hist_unw.append(phi_unw)
                        line_phi.set_data(t_hist, phi_hist_unw)
                        ax_res.set_xlim(max(0, t_now - PHASE_WINDOW_SEC), t_now); ax_res.set_ylim(-30, 30)
                        if rec_active and record_writer is not None:
                            row = [f"{t_now:.6f}", f"{phi:.9f}", f"{phi_unw:.9f}", f"{magZ:.9f}"] + [f"{val:.6f}" for val in yC]
                            record_writer.writerow(row)
                            if record_file: record_file.flush()
                            elapsed = int(time.time() - rec_start_time) if rec_start_time else 0
                            mm, ss = divmod(elapsed, 60); timer_text.set_text(f"{mm:02d}:{ss:02d}")
                fig.canvas.flush_events(); plt.pause(0.0005)
    finally:
        if record_file: record_file.flush(); record_file.close(); print(f"⏹ Saved {record_path}")
        sp.close()

if __name__=="__main__": main()

