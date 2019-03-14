import methods


vid, vid_length, vid_fps, vid_width, vid_length, vid_fourcc = methods.read_video('GP010016.MP4')

print(vid_length, vid_fps, vid_width, vid_length, vid_fourcc)

print("Script runs")