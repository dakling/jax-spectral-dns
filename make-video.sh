#!/usr/bin/env sh


# ffmpeg -y -f image2 -r 25 -pattern_type glob -i 'plots/plot_3d_z_vort_pertubation_y_t_*.png' -vcodec libx264 -crf 22 plots/vort_pert_y.mp4
# ffmpeg -y -f image2 -r 25 -pattern_type glob -i 'plots/plot_3d_z_velocity_pertubation_x_t_*.png' -vcodec libx264 -crf 22 plots/vel_pert_x.mp4
# ffmpeg -y -f image2 -r 25 -pattern_type glob -i 'plots/plot_3d_z_velocity_pertubation_y_t_*.png' -vcodec libx264 -crf 22 plots/vel_pert_y.mp4

# ffmpeg -y -f image2 -r 25 -pattern_type glob -i 'plots/plot_3d_z_vorticity_y_t_*.png' -vcodec libx264 -crf 22 plots/vort_y.mp4
# ffmpeg -y -f image2 -r 25 -pattern_type glob -i 'plots/plot_3d_z_velocity_x_t_*.png' -vcodec libx264 -crf 22 plots/vel_x.mp4
# ffmpeg -y -f image2 -r 25 -pattern_type glob -i 'plots/plot_3d_z_velocity_y_t_*.png' -vcodec libx264 -crf 22 plots/vel_y.mp4

ffmpeg -y -f image2 -r 25 -pattern_type glob -i 'plots/plot_cl_velocity_x_y_t_*.png' -vcodec libx264 -crf 22 plots/vel_y.mp4
