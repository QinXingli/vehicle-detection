from moviepy.editor import VideoFileClip
from svm_pipeline import *
from yolo_pipeline import *
from lane import *


def pipeline_yolo(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)

    return output


def pipeline_svm(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_svm(img_undist, img_lane_augmented, lane_info)

    return output


if __name__ == "__main__":

    demo = 2  # 1:image (YOLO and SVM), 2: video (YOLO Pipeline), 3: video (SVM pipeline)

    if demo == 1:
        filename = 'examples/hutanwangzhuang.jpg_no_crop_img.jpg'
        image = mpimg.imread(filename)

        #(1) Yolo pipeline
        yolo_result = pipeline_yolo(image)
        plt.figure()
        plt.imshow(yolo_result)
        plt.title('yolo pipeline', fontsize=30)

        #(2) SVM pipeline
        # draw_img = pipeline_svm(image)
        # fig = plt.figure()
        # plt.imshow(draw_img)
        # plt.title('svm pipeline', fontsize=30)
        plt.show()

    elif demo == 2:
        # YOLO Pipeline
        video_output = 'examples/zhangwang_2021.04.18.14.59.48.113_1-compress-flame-10-subclip-10-20-output.mp4'
        clip1 = VideoFileClip("examples/zhangwang_2021.04.18.14.59.48.113_1-compress-flame-10.mp4").subclip(10, 20)
        clip = clip1.fl_image(pipeline_yolo)
        clip.write_videofile(video_output, audio=False)

    else:
        # SVM pipeline
        video_output = 'examples/zhangwang_2021.04.18.14.59.48.113_1-compress-flame-10-clip-2part-output-svm.mp4'
        clip1 = VideoFileClip("examples/zhangwang_2021.04.18.14.59.48.113_1-compress-flame-10-clip-2part.mp4").subclip()
        clip = clip1.fl_image(pipeline_svm)
        clip.write_videofile(video_output, audio=False)


