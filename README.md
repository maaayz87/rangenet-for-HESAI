# rangenet for HESAI XT32 LiDAR

`roslaunch rangenet_pp ros1_rangenet.launch`  

**输入topic**  
`/hesai/pandar`  
**输出topic**  
`/label_pointcloud`      <-XYZI  
`/label_pointcloud_rgb`  <-XYZRGB  

**origin**  
XYZITR 4 4 4 4(float) 8(double) 2(uint16_t)  
offset 0 4 8 16! 24! 32 (34)  
**now**  
XYZIRT  
offset 0 4 8 16 20 28 (32)
