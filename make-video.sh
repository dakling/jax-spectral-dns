#!/usr/bin/env sh

make_video(){
    # ffmpeg -y -f image2 -r 6 -pattern_type glob -i "plots/plot_$1_t_*.png" -vcodec libx264 -crf 22 "img/$2.mp4"
    convert "plots/plot_$1_t_*.png" "img/$2.gif"
}

combine_three(){
    convert "$2.gif" -coalesce a-%04d.gif                         # separate frames of 1.gif
    convert "$3.gif" -coalesce b-%04d.gif                         # separate frames of 2.gif
    convert "$4.gif" -coalesce c-%04d.gif                         # separate frames of 2.gif
    for f in a-*.gif; do convert $f ${f/a/b} ${f/a/c} +append $f; done  # append frames side-by-side
    convert -loop 0 -delay 20 a-*.gif "img/$1.gif"               # rejoin frames
    rm a*.gif b*.gif c*.gif                                     #clean up
}

# make_video 3d_z_vorticity_z vort_z_Re_5000_jimenez1990
# make_video iso_z_vorticity_z vort_z_Re_5000_isolines_jimenez1990
# make_video streamlines_velocity_moving_frame velocity_moving_frame_Re_5000_streamlines_jimenez1990

combine_three ./img/Re_5000_jimenez_1990 ./img/velocity_moving_frame_Re_5000_streamlines_jimenez1990 ./img/vort_z_Re_5000_isolines_jimenez1990 ./img/vort_z_Re_5000_jimenez1990
