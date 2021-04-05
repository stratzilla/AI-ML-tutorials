#!/usr/bin/gnuplot -persist

set terminal png
set output "schwefel.png"
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
set xtics -500, 500, 500 font ",8"
set ytics -500, 500, 500 font ",8"
set xrange [-550:550]
set yrange [-550:550]
set zrange [0:2000]
set xyplane 0.1
unset colorbox
splot (2*418.9829) - (x*sin(sqrt(abs(x))) + y*sin(sqrt(abs(y))))