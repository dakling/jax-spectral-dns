#!/usr/bin/env sh

make_video(){
    mkdir img || echo
    # ffmpeg -y -f image2 -r 3 -pattern_type glob -i "plots/plot_$1_t_*.png" -vcodec libx264 -crf 22 "img/$2.mp4"
    convert "plots/plot_$1_t_*.png" "img/$2.gif"
}

combine_two(){
    mkdir img || echo
    convert "./img/$2.gif" -coalesce a-%04d.gif                         # separate frames of 1.gif
    convert "./img/$3.gif" -coalesce b-%04d.gif                         # separate frames of 2.gif
    for f in a-*.gif; do convert +append $f ${f/a/b} $f; done  # append frames side-by-side
    convert -loop 0 -delay 20 a-*.gif "img/$1.gif"               # rejoin frames
    # ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -i "img/$4.mp4" -filter_complex hstack=inputs=3 "img/$1.mp4"
}

combine_three(){
    mkdir img || echo
    convert "./img/$2.gif" -coalesce a-%04d.gif                         # separate frames of 1.gif
    convert "./img/$3.gif" -coalesce b-%04d.gif                         # separate frames of 2.gif
    convert "./img/$4.gif" -coalesce c-%04d.gif                         # separate frames of 2.gif
    for f in a-*.gif; do convert +append $f ${f/a/b} ${f/a/c} $f; done  # append frames side-by-side
    convert -loop 0 -delay 20 a-*.gif "img/$1.gif"               # rejoin frames
    # ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -i "img/$4.mp4" -filter_complex hstack=inputs=3 "img/$1.mp4"
}

combine_four(){
    mkdir img || echo
    convert "./img/$2.gif" -coalesce a-%04d.gif                         # separate frames of 1.gif
    convert "./img/$3.gif" -coalesce b-%04d.gif                         # separate frames of 2.gif
    convert "./img/$4.gif" -coalesce c-%04d.gif                         # separate frames of 2.gif
    convert "./img/$5.gif" -coalesce d-%04d.gif                         # separate frames of 2.gif
    for f in a-*.gif; do
        convert +append $f ${f/a/b} ${f/a/e}; # append frames side-by-side
        convert +append ${f/a/c} ${f/a/d} ${f/a/f}; # append frames side-by-side
        convert -append ${f/a/e} ${f/a/f} $f; # append frames on top of each other
    done
    convert -loop 0 -delay 20 a-*.gif "img/$1.gif"               # rejoin frames
    # rm a*.gif b*.gif c*.gif d*.gif e*.gif f*.gif                                    #clean up
    # ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -i "img/$4.mp4" -filter_complex hstack=inputs=3 "img/$1.mp4"
}


# for run_jimenez_1990

# make_video 3d_z_vorticity_z vort_z_Re_5000_jimenez1990
# make_video iso_z_vorticity_z vort_z_Re_5000_isolines_jimenez1990
# make_video streamlines_velocity_moving_frame velocity_moving_frame_Re_5000_streamlines_jimenez1990

# combine_three Re_5000_jimenez_1990 velocity_moving_frame_Re_5000_streamlines_jimenez1990 vort_z_Re_5000_isolines_jimenez1990 vort_z_Re_5000_jimenez1990


# for run_transient_growth

make_video cl_vel_0_x_y Re_3000_vel0_x
make_video cl_vel_0_y_y Re_3000_vel0_y
make_video 3d_z_velocity_x Re_3000_velocity_x
make_video 3d_z_velocity_y Re_3000_velocity_y
make_video 3d_z_vorticity_z Re_3000_vorticity_z
make_video energy Re_3000_energy

# combine_three Re_3000_transient_growth_nonlinear Re_3000_velocity_x Re_3000_velocity_y Re_3000_vorticity_z
combine_two Re_3000_transient_growth_initial Re_3000_vel0_x Re_3000_vel0_y
# combine_three Re_3000_transient_growth Re_3000_velocity_x Re_3000_velocity_y Re_3000_energy
combine_four Re_3000_transient_growth_with_vort Re_3000_velocity_x Re_3000_velocity_y Re_3000_vorticity_z Re_3000_energy

rm a*.gif b*.gif c*.gif d*.gif e*.gif f*.gif                                    #clean up
