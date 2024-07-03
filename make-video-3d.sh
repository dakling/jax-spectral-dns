#!/usr/bin/env sh


make_video_mp4(){
    mkdir img &> /dev/null
    scfmt="640:480"
    ffmpeg -y -f image2 -r 3 -pattern_type glob -i "plots/plot_$1_t_*.png" -vcodec libx264 -crf 22 "img/$2.mp4" &> /dev/null
}

combine_two_mp4(){
    mkdir img &> /dev/null
    scfmt="640:480"
    layout="0_0|w0_0"
    scale="[0:v] scale=$scfmt[a0]; [1:v] scale=$scfmt [a1]; [a0][a1]xstack=inputs=2:layout=$layout[v]"
    ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -i "img/$4.mp4" -i "img/$5.mp4" -filter_complex "$scale" -map "[v]" "img/$1.mp4" &> /dev/null
    ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -filter_complex hstack=inputs=2 "img/$1.mp4" &> /dev/null
}

combine_three_mp4(){
    mkdir img &> /dev/null
    scfmt="640:480"
    ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -i "img/$4.mp4"  -filter_complex "[0:v] scale=$scfmt[a0]; [1:v] scale=$scfmt [a1]; [2:v] scale=$scfmt [a2]; [a0][a1][a2]xstack=inputs=3:layout=0_0|w0_0|w0+w1_0[v]" -map "[v]" "img/$1.mp4" # &> /dev/null
}

combine_four_mp4(){
    mkdir img &> /dev/null
    scfmt="640:480"
    layout="0_0|w0_0|0_h0|w0_h0"
    scale="[0:v] scale=$scfmt[a0]; [1:v] scale=$scfmt [a1]; [2:v] scale=$scfmt [a2]; [3:v] scale=$scfmt [a3]; [a0][a1][a2][a3]xstack=inputs=4:layout=$layout[v]"
    ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -i "img/$4.mp4" -i "img/$5.mp4" -filter_complex "$scale" -map "[v]" "img/$1.mp4" &> /dev/null
}

combine_six_mp4(){
    mkdir img &> /dev/null
    scfmt="640:480"
    layout="0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0"
    scale="[0:v] scale=$scfmt [a0]; [1:v] scale=$scfmt [a1]; [2:v] scale=$scfmt [a2]; [3:v] scale=$scfmt [a3]; [4:v] scale=$scfmt [a4]; [5:v] scale=$scfmt [a5]; [a0][a1][a2][a3][a4][a5]xstack=inputs=6:layout=$layout[v]"
    ffmpeg -i "img/$2.mp4" -i "img/$3.mp4" -i "img/$4.mp4" -i "img/$5.mp4" -i "img/$6.mp4" -i "img/$7.mp4" -filter_complex "$scale" -map "[v]" "img/$1.mp4" &> /dev/null
}

make_video_gif(){
    mkdir img &> /dev/null
    convert "plots/plot_$1_t_*.png" "img/$2.gif"
}

combine_two_gif(){
    mkdir img &> /dev/null
    convert "./img/$2.gif" -coalesce a-%04d.gif                         # separate frames of 1.gif
    convert "./img/$3.gif" -coalesce b-%04d.gif                         # separate frames of 2.gif
    for f in a-*.gif; do convert +append $f ${f/a/b} $f; done  # append frames side-by-side
    convert -loop 0 -delay 20 a-*.gif "img/$1.gif"               # rejoin frames
    rm a*.gif b*.gif
}

combine_three_gif(){
    mkdir img &> /dev/null
    convert "./img/$2.gif" -coalesce a-%04d.gif                         # separate frames of 1.gif
    convert "./img/$3.gif" -coalesce b-%04d.gif                         # separate frames of 2.gif
    convert "./img/$4.gif" -coalesce c-%04d.gif                         # separate frames of 2.gif
    for f in a-*.gif; do convert +append $f ${f/a/b} ${f/a/c} $f; done  # append frames side-by-side
    convert -loop 0 -delay 20 a-*.gif "img/$1.gif"               # rejoin frames
    rm a*.gif b*.gif c*.gif
}

combine_four_gif(){
    mkdir img &> /dev/null
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

combine_six(){
    # if $GIF; then
    #     combine_six_gif $@
    # fi
    if $MP4; then
        combine_six_mp4 $@
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

rm img/* &> /dev/null

# adapt this to the specific case output
GIF=false
MP4=true


# initial condition evolution
make_video isosurfaces_vel_0_x __isosurfaces_vel_0_x
make_video isosurfaces_vel_0_y __isosurfaces_vel_0_y
make_video isosurfaces_vel_0_z __isosurfaces_vel_0_z
make_video phase_space __phase_space
make_video gain_over_iterations __gain_over_iterations
make_video localisation __localisation
combine_six initial_condition_evolution_isosurfaces __isosurfaces_vel_0_x __isosurfaces_vel_0_y __isosurfaces_vel_0_z __gain_over_iterations __phase_space __localisation

make_video 3d_z_vel_0_x __3d_z_vel_0_x
make_video 3d_z_vel_0_y __3d_z_vel_0_y
make_video 3d_z_vel_0_z __3d_z_vel_0_z
combine_six initial_condition_evolution_z __3d_z_vel_0_x __3d_z_vel_0_y __3d_z_vel_0_z __gain_over_iterations __phase_space __localisation

make_video 3d_x_vel_0_x __3d_x_vel_0_x
make_video 3d_x_vel_0_y __3d_x_vel_0_y
make_video 3d_x_vel_0_z __3d_x_vel_0_z
combine_six initial_condition_evolution_x __3d_x_vel_0_x __3d_x_vel_0_y __3d_x_vel_0_z __gain_over_iterations __phase_space __localisation

# final calculation
make_video isosurfaces_velocity_x __isosurfaces_velocity_x
make_video isosurfaces_velocity_y __isosurfaces_velocity_y
make_video isosurfaces_velocity_z __isosurfaces_velocity_z
make_video energy __energy
combine_four final_run_isosurfaces __isosurfaces_velocity_x __isosurfaces_velocity_y __isosurfaces_velocity_z __energy

make_video 3d_z_velocity_x __3d_z_velocity_x
make_video 3d_z_velocity_y __3d_z_velocity_y
make_video 3d_z_velocity_z __3d_z_velocity_z
combine_four final_run_3d_z __3d_z_velocity_x __3d_z_velocity_y __3d_z_velocity_z __energy

make_video 3d_x_velocity_x __3d_x_velocity_x
make_video 3d_x_velocity_y __3d_x_velocity_y
make_video 3d_x_velocity_z __3d_x_velocity_z
combine_four final_run_3d_x __3d_x_velocity_x __3d_x_velocity_y __3d_x_velocity_z __energy

cleanup
