#!/usr/bin/env sh

make_video(){
    # ffmpeg -y -f image2 -r 6 -pattern_type glob -i "plots/plot_$1_t_*.png" -vcodec libx264 -crf 22 "img/$2.mp4"
    convert "plots/plot_$1_t_*.png" "img/$2.gif"
}

make_video 3d_z_vorticity_z vort_z_Re_5000_jimenez1990
make_video iso_z_vorticity_z vort_z_Re_5000_isolines_jimenez1990
make_video streamlines_velocity_moving_frame velocity_moving_frame_Re_5000_streamlines_jimenez1990
