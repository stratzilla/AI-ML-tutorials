#!/usr/bin/gnuplot -persist

set terminal png
set output "styb-tang-map.png"
set pm3d map
set palette rgb 31,13,32
set isosamples 2000
set xlabel "x_1" font ",10"
set ylabel "x_2" font ",10"
set xtics -5, 1, 5 font ",8"
set ytics -5, 1, 5 font ",8"
set xrange [-5:5]
set yrange [-5:5]
unset key
set lmargin screen 0.10
set rmargin screen 0.85
set bmargin screen 0.15
set tmargin screen 0.95
set autoscale xfix
set autoscale yfix
splot (0.5 * ((x**4 - 16*(x**2) + 5*x) + (y**4 - 16*(y**2) + 5*y)))