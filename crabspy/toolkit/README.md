Crabspy Tools
===================

Useful tools for processing videos and images.

## Content:

- [Listing all video files](#Listing-all-video-files)
- [3D models workflow](#3D-models-workflow)

## Listing all video files
```
python list_videos.py
```

Module for recursively listing all videos in a folder and extracting meta information associated to videos

Requirements:

- ffmpeg

The resulting file would look like:

| file_name	| file_directory | file_extension | error | file_path | file_size | file_created | tag_major_brand | file_duration | duration_ts | nb_frames | fps | codec | codec_tag | aspect_ratio | width | height |
| --------- | -------------- | -------------- | ----- | --------- | --------- | ------------ | --------------- | ------------- | ------------| --------- | --- | ----- | --------- | ------------ | ----- | ------ |
|VIRB0028	|E:\Crabspy-VideoZoo	|.MP4	|FALSE	|E:\Crabspy-VideoZoo\VIRB0028.MP4	|125829120	|2017-10-13T10:30:49.000000Z	|avc1	|00:34.7	|8328320	|1040	|30000/1001	|h264	|0x31637661	|4:03	|1920	|1440
|VIRB0016	|E:\Crabspy-VideoZoo	|.MP4	|FALSE	|E:\Crabspy-VideoZoo\VIRB0016.MP4	|2831155200	|2019-09-22T08:58:34.000000Z	|avc1	|15:00.1	|216023808	|26976	|30000/1001	|h264	|0x31637661	|4:03	|1920	|1440
|VIRB0002	|E:\Crabspy-VideoZoo	|.MP4	|FALSE	|E:\Crabspy-VideoZoo\VIRB0002.MP4	|83886080	|2017-09-27T13:31:57.000000Z	|avc1	|00:20.7	|4968964	|1241	|60000/1001	|h264	|0x31637661	|16:09	|1920	|1080
|VIRB0015   |E:\Crabspy-VideoZoo	|.MP4	|TRUE	| Command '['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-sexagesimal', '-show_streams', 'E:\\Crabspy-VideoZoo\\VIRB0015.MP4']' returned non-zero exit status 1| | | | | | | | | | |
|VIRB0048   |E:\Crabspy-VideoZoo	|.MP4	|TRUE	|KeyError: 'display_aspect_ratio' | | | | | | | | | | |

Please observe that this command will not populate all columns in the resulting CSV file if:
  * files are corrupted
  * files cannot be opened
  * or, where ```ffprobe``` keys do not exist in multimedia streams associated to the file

## 3D models workflow 

More information in the [3Dmodels](https://github.com/CexyNature/Crabspy/tree/master/crabspy/toolkit/3Dmodels) folder