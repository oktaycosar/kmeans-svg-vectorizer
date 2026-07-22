"""Vector Magic-style vectorizer: K-Means → Mask → Boundary → SVG path"""
import cv2, numpy as np, time

t0=time.time()
img=cv2.imread('test logo 3.png')
bgr=img[:,:,:3] if img.shape[2]==4 else img
h,w=bgr.shape[:2]

t1=time.time()
# 1. Classify + Palette: K-Means
px=bgr.reshape(-1,3).astype(np.float32)
k=12
_,labels,centers=cv2.kmeans(px,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,15,1.0),5,cv2.KMEANS_PP_CENTERS)
centers=centers.astype(np.uint8)
print(f'Palette: {k} colors, {time.time()-t1:.2f}s')

t2=time.time()
# 2. Segment: extract masks per color
masks=[]
for ci in range(k):
    mask=(labels.flatten()==ci).reshape(h,w).astype(np.uint8)*255
    if np.count_nonzero(mask)>=50:
        masks.append((ci,mask))
print(f'Segment: {len(masks)} regions, {time.time()-t2:.2f}s')

t3=time.time()
# 3. Smooth + Fit: contours → simplified paths
paths=[]
for ci,mask in masks:
    # Morph close to smooth
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    color=centers[ci]
    hex=f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}'
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area<30: continue
        peri=cv2.arcLength(cnt,True)
        eps=0.0015*peri  # Aggressive simplification
        approx=cv2.approxPolyDP(cnt,eps,True)
        pts=approx.reshape(-1,2)
        if len(pts)<3: continue
        d='M '+' L '.join(f'{x:.1f} {y:.1f}' for x,y in pts)+' Z'
        paths.append(f'<path d="{d}" fill="{hex}" stroke="{hex}" stroke-width="0.5" stroke-linejoin="round"/>')
print(f'Fit: {len(paths)} paths, {time.time()-t3:.2f}s')

# 4. Output SVG
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
svg.append(f'<rect width="{w}" height="{h}" fill="white"/>')
svg.extend(paths)
svg.append('</svg>')
open('logo_output3/vm_style.svg','w').write('\n'.join(svg))
print(f'Total: {time.time()-t0:.2f}s → logo_output3/vm_style.svg ({len(paths)} paths)')
