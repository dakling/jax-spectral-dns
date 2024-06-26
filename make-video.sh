#!/usr/bin/env sh


make_video_mp4(){
    mkdir img || echo
    ffmpeg -y -f image2 -r 3 -pattern_type glob -i "plots/plot_$1_t_*.png" -vcodec libx264 -crf 22 "img/$2.mp4" &> /dev/null
}

combine_two_mp4(){
    mkdir img || echo
    ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -filter_complex hstack=inputs=2 "img/$1.mp4" &> /dev/null
}

combine_three_mp4(){
    mkdir img || echo
    ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -i "img/$4.mp4" -filter_complex hstack=inputs=3 "img/$1.mp4" &> /dev/null
}

combine_four_mp4(){
    mkdir img || echo
    ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -i "img/$4.mp4" -i "img/$5.mp4" -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" "img/$1.mp4" &> /dev/null
}

combine_five_mp4(){
    mkdir img || echo
    ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -i "img/$4.mp4" -i "img/$5.mp4" -i "img/$6.mp4" -filter_complex hstack=inputs=5 "img/$1.mp4" &> /dev/null
}

make_video_gif(){
    mkdir img || echo
    convert "plots/plot_$1_t_*.png" "img/$2.gif"
}

combine_two_gif(){
    mkdir img || echo
    convert "./img/$2.gif" -coalesce a-%04d.gif                         # separate frames of 1.gif
    convert "./img/$3.gif" -coalesce b-%04d.gif                         # separate frames of 2.gif
    for f in a-*.gif; do convert +append $f ${f/a/b} $f; done  # append frames side-by-side
    convert -loop 0 -delay 20 a-*.gif "img/$1.gif"               # rejoin frames
    rm a*.gif b*.gif
}

combine_three_gif(){
    mkdir img || echo
    convert "./img/$2.gif" -coalesce a-%04d.gif                         # separate frames of 1.gif
    convert "./img/$3.gif" -coalesce b-%04d.gif                         # separate frames of 2.gif
    convert "./img/$4.gif" -coalesce c-%04d.gif                         # separate frames of 2.gif
    for f in a-*.gif; do convert +append $f ${f/a/b} ${f/a/c} $f; done  # append frames side-by-side
    convert -loop 0 -delay 20 a-*.gif "img/$1.gif"               # rejoin frames
    rm a*.gif b*.gif c*.gif
}

combine_four_gif(){
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
    rm a*.gif b*.gif c*.gif d*.gif e*.gif f*.gif                                    #clean up
}

make_video(){
    if $GIF; then
        make_video_gif $@
    fi
    if $MP4; then
        make_video_mp4 $@
    fi
}

combine_two(){
    if $GIF; then
        combine_two_gif $@
    fi
    if $MP4; then
        combine_two_mp4 $@
    fi
}

combine_three(){
    if $GIF; then
        combine_three_gif $@
    fi
    if $MP4; then
        combine_three_mp4 $@
    fi
}

combine_four(){
    if $GIF; then
        combine_four_gif $@
    fi
    if $MP4; then
        combine_four_mp4 $@
    fi
}

combine_five(){
    # if $GIF; then
    #     combine_five_gif $@
    # fi
    if $MP4; then
        combine_five_mp4 $@
    fi
}

cleanup(){
    if $GIF; then
        rm img/__*.gif
    fi
    if $MP4; then
        rm img/__*.mp4
    fi
}

rm img/*

# adapt this to the specific case output
GIF=false
MP4=true


make_video 3d_z_vel_0_x __vel0_z_3d_x
make_video 3d_z_vel_0_y __vel0_z_3d_y
make_video 3d_z_vel_0_z __vel0_z_3d_z
make_video 3d_x_vel_0_x __vel0_x_3d_x
make_video 3d_x_vel_0_y __vel0_x_3d_y
make_video 3d_x_vel_0_z __vel0_x_3d_z
make_video 3d_z_velocity_x __velocity_z_x
make_video 3d_z_velocity_y __velocity_z_y
make_video 3d_z_vorticity_z __vorticity_z_z
make_video 3d_x_velocity_x __velocity_x_x
make_video 3d_x_velocity_y __velocity_x_y
make_video 3d_x_vorticity_z __vorticity_x_z
make_video energy __energy
make_video phase_space __phase_space
make_video gain_over_iterations __gain_over_iterations

combine_five transient_growth_initial_x __vel0_x_3d_x __vel0_x_3d_y __vel0_x_3d_z __phase_space __gain_over_iterations
combine_five transient_growth_initial_z __vel0_z_3d_x __vel0_z_3d_y __vel0_z_3d_z __phase_space __gain_over_iterations
combine_four transient_growth_z __velocity_z_x __velocity_z_y __vorticity_z_z __energy

cleanup
