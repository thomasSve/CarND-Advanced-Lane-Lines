from image_gen import process_img
from moviepy.editor import VideoFileClip

def main(input_video = 'project_video.mp4', output_video = 'output_video.mp4'):
    clip1 = VideoFileClip(input_video)
    video_clip = clip1.fl_image(process_img)
    video_clip.write_videofile(output_video, audio = False)
    

if __name__ == '__main__':
    main()
