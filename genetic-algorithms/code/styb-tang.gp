#!/usr/bin/gnuplot -persist

set terminal png
set output "styb-tang.png"
set xlabel "x_1" font ",10"
set ylabel "x_2" font ",10"
unset border
set border 1+2+4+8+16+32+64
set hidden3d
set pm3d
set palette rgb 31,13,32
set isosamples 250
set view 60, 45, 1, 1
unset key
set xtics -5, 5, 5 font ",8"
set ytics -5, 5, 5 font ",8"
set xrange [-5:5]
set yrange [-5:5]
set zrange [-100:250]
set xyplane 0.1
unset colorbox
splot (0.5 * ((x**4 - 16*(x**2) + 5*x) + (y**4 - 16*(y**2) + 5*y)))