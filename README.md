# Camera Calibration 2022

*These are Camera Calibration codes for DFX 3D Scanner produced by Wang.ZiHao in 2022. We temp to refine the camera params by ourselves through several cost functions and regularizations.*

*We found that 3D cost functions and regularization is needed in order to improve 3D reconstruction accuracy.*

## Released Cost functions and regularizations

*In previous works we utilized Scipy to do the least square optimizing but quite little improvement could be find by our eyes so we decided to do the whole optimizing in Pytorch.*

*Now the Pytorch version supports BatchSize and Dataloader -2022.2.11*

### Single Camera

- [] 2D cost function

- [] 3D cost function

### Stereo Camera

- [] 2D cost function

- [] 3D cost function with z=0 regularization

- [x] Fitting the background planes *(Only regularization for plane)*
- 
- [x] Fitting the background planes with Triangulation loss 
- 




