#!/usr/bin/gnuplot -persist

set terminal png
set output "schwefel-map.png"
set pm3d map
set palette rgb 31,13,32
set isosamples 2000
set xlabel "x_1" font ",10"
set ylabel "x_2" font ",10"
set xtics -500, 100, 500 font ",8"
set ytics -500, 100, 500 font ",8"
set xrange [-500:500]
set yrange [-500:500]
unset key
set lmargin screen 0.12
set rmargin screen 0.85
set bmargin screen 0.15
set tmargin screen 0.95
set autoscale xfix
set autoscale yfix
splot (2*418.9829) - (x*sin(sqrt(abs(x))) + y*sin(sqrt(abs(y))))