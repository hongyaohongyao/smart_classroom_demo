import torch

torch._C._jit_set_profiling_mode(False)
torch.jit.optimized_execution(False)

yolov5_weight = './weights/yolov5s.torchscript.pt'
alphapose_weight = './weights/halpe136_mobile.torchscript.pth'
device = 'cuda'

if __name__ == '__main__':
    # TaskSolution() \
    #     .set_source_module(VideoModule()) \
    #     .set_next_module(YoloV5Module(yolov5_weight, device)) \
    #     .set_next_module(AlphaPoseModule(alphapose_weight, device)) \
    #     .set_next_module(DrawModule()) \
    #     .start()
    # img = cv2.imread('resource/pic/human_boarder.png')
    # print(img[img == 253])
    
    while True:
        for t in range(1, m):
            if (m - t) * (m - t - 1) / (m * (m - 1)) == 0.5:
                print(f"m:{m}, t:{t}")
        m += 1
