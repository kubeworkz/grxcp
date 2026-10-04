"""
The ADC survey, as far as the tile needs it: what a column's converter costs.

pta_power.py priced the tile's ADCs from three converters taken one at a time
from their papers, and said that B. Murmann's ADC Performance Survey was the
place to widen that.  This is the survey, from the spreadsheet its author
keeps: every converter published at ISSCC and the VLSI Symposium from 1997 to
2026, 764 operating points, of which 763 give a power, a rate and an SNDR.

    B. Murmann, "ADC Performance Survey 1997-2026," [Online].
    Available: https://github.com/bmurmann/ADC-survey

THE QUESTION PUT TO IT.  A column needs a converter of at least B effective
bits that samples at the shot rate or faster.  Any published part that does
both can do the job and no other can, so the price is taken from those parts:
the energy each spends on a sample, P / fs, cheapest first.  Two figures are
kept, the best and the fifth-best -- five because the survey's own envelope is
drawn through its five best points.  A part faster than the shot rate is priced
at the shot rate by its energy a sample, which ASSUMES its power follows its
clock down; as_published() is the cheapest part run exactly as it was measured,
and assumes nothing.

WHAT THIS REPLACED.  The survey was first read here from the plots of its
author's ISSCC 2022 short course, which stop at 2021 and carry no names, and
priced by the figure of merit of the parts "near" a rate: within a factor of
2.5 of it, between 30 and 50 dB, the best and the worst of the leading quarter.
The spreadsheet shows three things wrong with that, and the third is the
method's and not the slides':

  - it was five years short.  The best 7-bit converter at 1 GS/s or more was
    2.55 pJ a sample in 2021 and is 1.11 now
  - the leading quarter was a count of the parts that were missing.  The
    slides' third plot starts at 130 dB of FoM_S, so the three-way match never
    saw the least efficient converters: 9 of the 43 near 1 GS/s.  With them
    the quarter mark is 86 fJ a step, not 40
  - "near" is not the job.  A 4.8 GS/s part is near 10 GS/s and cannot sample
    at it, and a 31 dB part is near 1 GS/s and is not a 7-bit converter.  What
    ten times the rate costs came out at fifteen to sixteen times the power
    that way.  From the parts that can do it, it is over fifty

HOW THE TWO READINGS AGREE.  The slides' plots are vector drawings, and 546
converters were matched across three of them (read_slides(), below).  Held to
the spreadsheet's rows to 2021, 541 are there, one to one, to 0.04 dB and 0.1%
in power.  The other five are not converters: each is three different parts'
markers that happen to agree across the three plots.  None of the five is at
6 bits and 0.1 GS/s, so none was in a figure the tile used.  And 77 of the
spreadsheet's 618 rows to 2021 the slides never gave, 43 of them below that
130 dB.

WHAT IS KEPT HERE.  Every operating point of at least 6 effective bits
(37.88 dB) at 0.1 GS/s and above: year, conference, paper, architecture,
process in um as the survey gives it, Nyquist rate in GS/s (the survey's
fsnyq), SNDR in dB (its SNDR_plot, measured near Nyquist where the paper gave
that), power in mW.  A paper can have more than one row.  These are the
survey's data and not a new measurement of anything, and the papers were not
read here: where a title states a figure, as "3-mW 2.7-GS/s" and
"9.8-fJ/conv.-step" do, the row agrees with it.

The parts the figures lean on, as the survey lists them:
    ISSCC 2023 17.7   J.-C. Wang and T.-H. Kuo, "A 3-mW 2.7-GS/s 8b Subranging
                      ADC with Multiple-Reference-Embedded Comparators", 28 nm
    VLSI 2022 C10-4   J.-C. Wang, B.-Y. Li and T.-H. Kuo, "A 9.8-fJ/conv.-step
                      FoMW 8b 2.5-GS/s Single-Channel CDAC-Assisted Subranging
                      ADC with Reference-Embedded Comparators", 28 nm
    ISSCC 2024 22.4   Y. Tao et al., "A 4.8GS/s 7-ENoB Time-Interleaved SAR ADC
                      ...", 28 nm; the survey notes its power is the core's
    VLSI 2020 CD1.3   D.-R. Oh et al., "An 8b 1GS/s 2.55mW SAR-Flash ADC with
                      Complementary Dynamic Amplifiers", 28 nm
    VLSI 2026 C17.3   C. Kim et al., "An 8bit 4.8GS/s 10.9fJ/conv.-step TI-SAR
                      ADC ...", 14 nm
    ISSCC 2025 24.7   Y. Tao et al., "An 8b 10GS/s 2-Channel Time-Interleaved
                      Pipelined ADC ...", 28 nm
    VLSI 2024 C24.2   J. Liu et al., "A 16GS/s 10b Time-domain ADC using
                      Pipelined-SAR TDC ... in 4nm CMOS"

WHAT IT IS NOT.  The power is the converter's as its paper reported it, which
usually leaves out the reference, the clock and whatever drives the input.
These are conference papers, each the best its authors could show, and not
parts that can be bought.  And the best of them are recent: three of the four
cheapest 7-bit converters were published after the slides were drawn.

With the spreadsheet as an argument this re-reads it and holds the table to
it.  With the slides' PDF it re-reads the plots and holds them to the table.
With both it holds the two to each other.  Without, it reports from the table.

Standard library only.
Run:  python3 docs/designs/pta_adc_survey.py [ADCsurvey.ods] [SLIDES.pdf]
"""
import math
import re
import sys
import zipfile
import zlib
import xml.etree.ElementTree as ET

YEAR, VENUE, ID, ARCH, NODE, FS, SNDR, P = range(8)

# year, conference, paper, architecture, process (um), fs in GS/s, SNDR in dB, P in mW
TABLE = (
    (1997, "ISSCC", "13.4", "SDCT, BP", "HBT", 0.1252, 44.1, 1400.0),
    (1997, "ISSCC", "8.4", "Two-step", "Bipolar", 0.128, 58.4, 5700.0),
    (1998, "VLSI", "14.3", "Two-Step, Pipe", "0.25 BiCMOS", 0.1, 43.3, 180.0),
    (1999, "VLSI", "8.2", "Pipe", "Bipolar", 0.5, 45.0, 950.0),
    (1999, "VLSI", "8.4", "Pipe, Folding", "0.5", 0.1, 42.3, 165.0),
    (2000, "ISSCC", "2.1", "Folding", "Bipolar", 0.1, 75.0, 1250.0),
    (2000, "VLSI", "16.1", "Pipe", "0.6 BiCMOS", 0.105, 68.0, 850.0),
    (2000, "VLSI", "16.2", "Folding", "0.35", 0.125, 40.2, 110.0),
    (2001, "ISSCC", "8.3", "Pipe", "0.18", 0.1, 57.12, 180.0),
    (2002, "ISSCC", "10.1", "Pipe, TI", "0.35", 4.0, 39.08, 4600.0),
    (2002, "ISSCC", "10.4", "Pipe, TI", "0.35", 0.12, 56.8, 234.0),
    (2002, "VLSI", "23.2", "Folding", "0.12", 0.1, 54.92, 180.0),
    (2003, "ISSCC", "18.5", "Pipe", "0.18", 0.15, 52.0, 123.0),
    (2004, "ISSCC", "14.1", "Folding, TI", "0.18", 1.6, 45.8, 1269.0),
    (2004, "ISSCC", "14.2", "Folding", "0.18", 0.6, 46.91, 200.0),
    (2004, "ISSCC", "14.3", "Pipe", "0.13", 0.22, 51.0, 135.0),
    (2004, "ISSCC", "14.4", "Pipe, TI", "0.18", 0.15, 45.4, 71.0),
    (2004, "ISSCC", "14.5", "Two-Step", "0.13", 0.125, 47.51, 21.0),
    (2004, "VLSI", "25.2", "Folding", "0.18", 0.6, 40.0, 207.0),
    (2004, "VLSI", "25.3", "Pipe", "0.34 BiCMOS", 0.18, 60.39, 1200.0),
    (2005, "ISSCC", "15.4", "Pipe", "0.18", 0.125, 53.7, 40.0),
    (2005, "ISSCC", "15.5", "Pipe", "0.18", 0.2, 47.3, 30.0),
    (2006, "ISSCC", "12.6", "Pipe", "0.13", 0.1, 66.0, 224.0),
    (2006, "ISSCC", "31.6", "Pipe, TI", "0.13", 1.0, 52.0, 250.0),
    (2006, "VLSI", "16.2", "Pipe, SwOpAmp", "0.18", 0.1, 41.5, 30.0),
    (2007, "ISSCC", "25.2", "Two-Step", "0.09", 0.16, 56.54, 84.0),
    (2007, "ISSCC", "25.4", "Pipe", "0.09", 0.205, 55.0, 61.0),
    (2007, "ISSCC", "25.5", "Pipe, ZCBC", "0.18", 0.2, 40.29, 8.5),
    (2007, "ISSCC", "25.6", "Pipe", "0.13", 0.205, 54.0, 92.5),
    (2007, "ISSCC", "25.7", "Pipe, TI", "0.09", 0.8, 54.0, 350.0),
    (2007, "VLSI", "7.1", "SAR, TI", "0.13", 1.35, 48.11, 168.4),
    (2007, "VLSI", "7.3", "Pipe, TI", "0.09", 1.1, 40.89, 46.0),
    (2008, "ISSCC", "12.3", "Two-Step", "0.09", 0.15, 40.0, 0.1335),
    (2008, "ISSCC", "12.6", "Pipe", "0.18", 0.1, 72.2, 230.0),
    (2008, "ISSCC", "12.7", "Pipe", "0.065", 0.1, 59.0, 4.5),
    (2008, "ISSCC", "30.4", "Pipe", "0.065", 0.2, 59.9, 180.0),
    (2008, "ISSCC", "30.7", "Two-Step", "0.09", 0.3, 38.4, 34.0),
    (2008, "VLSI", "22.4", "Pipe", "0.18", 0.25, 65.9, 140.0),
    (2008, "VLSI", "8.1", "Pipe, TI", "0.065", 0.8, 44.2, 300.0),
    (2008, "VLSI", "8.2", "Pipe", "0.09", 0.1, 70.0, 250.0),
    (2009, "ISSCC", "4.3", "Folding, TI", "0.18", 0.9955, 56.5, 1200.0),
    (2009, "ISSCC", "4.5", "SAR, TI", "0.13", 0.6, 43.0, 30.0),
    (2009, "ISSCC", "4.6", "Pipe", "0.09", 0.5, 53.0, 55.0),
    (2009, "ISSCC", "4.7", "Pipe", "0.18", 0.125, 77.0, 385.0),
    (2009, "ISSCC", "9.1", "Pipe", "0.09", 0.1, 68.8, 130.0),
    (2009, "VLSI", "23.1", "SAR", "0.13", 0.1, 52.0, 0.92),
    (2010, "ISSCC", "16.1", "Pipe", "0.18 BiCMOS", 0.25, 77.5, 850.0),
    (2010, "ISSCC", "16.2", "Pipe", "0.18 BiCMOS", 0.16, 75.2, 1620.0),
    (2010, "ISSCC", "16.5", "Pipe", "0.09", 0.1, 53.9, 4.5),
    (2010, "ISSCC", "21.5", "SAR", "0.065", 0.1, 56.0, 1.13),
    (2010, "VLSI", "23.1", "Pipe, ZCBC", "0.09", 0.1, 63.16, 6.2),
    (2010, "VLSI", "23.4", "SAR", "0.09", 0.15, 54.07, 1.53),
    (2011, "ISSCC", "10.1", "SAR, TI", "0.065", 2.6, 48.5, 480.0),
    (2011, "ISSCC", "10.2", "Pipe, TI", "0.18 BiCMOS", 1.0, 59.0, 575.0),
    (2011, "ISSCC", "10.3", "Pipe, TI", "0.04", 0.8, 59.0, 105.0),
    (2011, "ISSCC", "10.5", "SAR", "0.065", 0.4, 40.4, 4.0),
    (2011, "ISSCC", "27.1", "SDCT", "0.045", 0.25, 65.0, 256.0),
    (2011, "VLSI", "12.1", "Pipe, TI", "0.04", 3.0, 51.0, 500.0),
    (2011, "VLSI", "12.2", "Pipe", "0.04", 0.3, 56.0, 40.0),
    (2011, "VLSI", "12.4", "Pipe", "0.09", 0.32, 50.0, 40.0),
    (2011, "VLSI", "12.5", "Subranging", "0.055", 1.0, 40.0, 16.0),
    (2011, "VLSI", "25.4", "Single-Slope, TI", "0.13", 1.0, 38.9, 26.5),
    (2011, "VLSI", "4.4", "SDCT, BP", "0.04", 0.16, 41.0, 163.5),
    (2012, "ISSCC", "27.3", "Pipe", "0.065", 0.2, 57.0, 5.37),
    (2012, "ISSCC", "27.5", "Pipe", "0.04", 0.25, 56.0, 1.7),
    (2012, "ISSCC", "8.3", "SDCT, BP", "0.065", 0.3, 74.0, 550.0),
    (2012, "ISSCC", "8.7", "SDCT", "0.045", 0.12, 60.6, 20.0),
    (2012, "VLSI", "11.1", "SAR, TI", "0.065", 2.8, 48.2, 44.6),
    (2012, "VLSI", "11.2", "SAR TI", "0.065", 1.0, 42.75, 3.8),
    (2012, "VLSI", "11.3", "SAR", "0.028", 0.75, 43.3, 4.5),
    (2012, "VLSI", "11.4", "Pipe, TI", "0.065", 0.5, 52.94, 8.2),
    (2012, "VLSI", "4.3", "Pipe", "0.065", 1.0, 52.4, 32.9),
    (2013, "ISSCC", "26.2", "SAR, TI", "0.065", 3.6, 50.0, 795.0),
    (2013, "ISSCC", "26.3", "Pipe, TI", "0.13 BiCMOS", 2.5, 61.0, 23900.0),
    (2013, "ISSCC", "26.4", "SAR", "0.032", 1.2, 39.3, 3.06),
    (2013, "ISSCC", "26.5", "SAR, TI", "0.045", 0.9, 51.2, 10.8),
    (2013, "VLSI", "21.5", "Pipe", "0.028", 0.41, 55.0, 2.14),
    (2013, "VLSI", "8.1", "Pipe, TI", "0.028", 5.4, 50.0, 500.0),
    (2013, "VLSI", "8.4", "Pipe", "0.065", 0.2, 57.6, 11.5),
    (2013, "VLSI", "8.5", "Pipe", "0.065", 0.8, 52.2, 19.0),
    (2014, "ISSCC", "11.5", "Pipe", "0.065", 0.1, 56.3, 2.46),
    (2014, "ISSCC", "22.4", "SAR, TI", "0.065", 1.0, 51.4, 18.9),
    (2014, "ISSCC", "22.5", "SAR, TI", "0.04", 1.62, 48.0, 93.0),
    (2014, "ISSCC", "29.3", "Pipe", "0.065", 1.0, 68.0, 1200.0),
    (2014, "VLSI", "10.1", "SDCT, VCO", "0.065", 0.1, 64.0, 38.0),
    (2014, "VLSI", "10.2", "SDCT", "0.02", 0.16, 67.5, 23.0),
    (2014, "VLSI", "23.1", "Pipe, SAR, TI", "0.028", 0.2, 65.0, 2.3),
    (2014, "VLSI", "23.2", "Pipe, SAR", "0.065", 0.21, 60.1, 5.3),
    (2014, "VLSI", "23.3", "SAR", "0.028", 0.1, 67.1, 8.0),
    (2014, "VLSI", "23.4", "Pipe, SAR", "0.04", 0.16, 65.3, 4.96),
    (2015, "ISSCC", "15.1", "SDCT, SAR", "0.028", 0.1, 74.6, 78.0),
    (2015, "ISSCC", "15.6", "Pipe", "0.065", 0.25, 65.7, 49.7),
    (2015, "ISSCC", "15.8", "Pipe", "0.18", 0.5, 64.0, 550.0),
    (2015, "ISSCC", "26.3", "Pipe, TI", "0.028", 0.8, 57.14, 76.4),
    (2015, "ISSCC", "26.4", "SAR, TI", "0.045", 1.6, 56.1, 17.3),
    (2015, "ISSCC", "26.7", "Pipe, SAR, TI", "0.028", 5.0, 46.1, 150.0),
    (2015, "ISSCC", "26.7", "SAR, TI", "0.045", 1.7, 51.2, 15.4),
    (2015, "VLSI", "18.1", "SDCT", "0.04", 0.15, 64.9, 22.8),
    (2015, "VLSI", "18.2", "SDCT, VCO", "0.065", 0.1, 71.5, 54.0),
    (2015, "VLSI", "3.5", "SAR, VCO", "0.045", 0.2, 68.0, 3.4),
    (2016, "ISSCC", "15.5", "SDCT", "0.028", 0.7, 64.8, 756.0),
    (2016, "ISSCC", "15.6", "SDCT", "0.016", 0.32, 65.33, 40.0),
    (2016, "ISSCC", "27.4", "Single-slope, SAR", "0.028", 0.1, 64.43, 0.35),
    (2016, "ISSCC", "27.5", "Pipe, TI", "0.065", 4.0, 55.5, 2200.0),
    (2016, "ISSCC", "27.6", "Pipe, TI", "0.016", 4.0, 56.0, 300.0),
    (2016, "ISSCC", "27.7", "SAR, TI", "0.04", 2.6, 50.6, 18.4),
    (2016, "ISSCC", "27.8", "Pipe, SAR, TI", "0.028", 0.6, 58.0, 26.5),
    (2016, "VLSI", "15.1", "SAR, TI", "0.065", 1.6, 65.0, 37.7),
    (2016, "VLSI", "15.2", "Pipe, SAR, TI", "0.028", 0.8, 60.8, 14.6),
    (2016, "VLSI", "19.2", "SAR, TI", "0.016", 1.6, 50.3, 8.2),
    (2016, "VLSI", "19.3", "Pipe, TI", "0.028", 5.0, 58.0, 2300.0),
    (2016, "VLSI", "3.4", "SDCT", "0.065", 0.12, 67.6, 13.3),
    (2017, "ISSCC", "16.1", "Pipe, SAR, TI", "0.016", 4.0, 57.3, 513.0),
    (2017, "ISSCC", "16.2", "Pipe", "0.028", 2.0, 66.0, 2330.0),
    (2017, "ISSCC", "16.4", "SAR", "0.028", 2.4, 40.05, 5.0),
    (2017, "ISSCC", "16.5", "SAR, TI", "0.028", 8.0, 49.0, 300.0),
    (2017, "ISSCC", "16.7", "Pipe, TI", "0.028", 10.0, 55.0, 2900.0),
    (2017, "ISSCC", "28.3", "SDCT, VCO", "0.016", 0.25, 71.9, 54.0),
    (2017, "ISSCC", "28.4", "Pipe, SAR", "0.065", 0.33, 63.5, 6.23),
    (2017, "ISSCC", "28.5", "Pipe, SAR", "0.014", 1.5, 50.1, 6.92),
    (2017, "ISSCC", "28.7", "Pipe, SAR", "0.028", 0.16, 61.1, 1.9),
    (2017, "VLSI", "21.1", "SAR", "0.014", 0.3, 60.45, 3.3),
    (2017, "VLSI", "21.2", "SAR, TI", "0.016", 2.0, 50.1, 10.4),
    (2017, "VLSI", "21.5", "Flash", "0.065", 2.0, 40.7, 21.0),
    (2017, "VLSI", "3.3", "SDCT", "0.04", 0.312, 64.0, 233.0),
    (2017, "VLSI", "8.1", "Pipe, SAR, TI", "0.016", 0.303, 64.0, 3.6),
    (2017, "VLSI", "8.2", "Pipe", "0.028", 0.5, 56.6, 6.0),
    (2017, "VLSI", "8.3", "Pipe", "0.028", 0.6, 56.3, 14.2),
    (2017, "VLSI", "8.4", "Pipe, SAR", "0.04", 0.1, 73.2, 2.3),
    (2018, "ISSCC", "14.1", "SDCT", "0.028", 0.1, 79.8, 64.3),
    (2018, "VLSI", "9.3", "Subranging, Time-Based", "0.065", 0.95, 45.0, 2.3),
    (2018, "VLSI", "9.4", "Two-step, SAR", "0.04", 1.1, 45.0, 4.0),
    (2018, "VLSI", "9.5", "SAR, TI", "0.016", 5.0, 57.0, 641.0),
    (2019, "ISSCC", "20.1", "VCO, TI", "0.028", 5.0, 45.2, 22.7),
    (2019, "ISSCC", "20.3", "SAR, TI", "0.04", 0.1, 70.4, 13.0),
    (2019, "ISSCC", "20.5", "SDCT", "0.028", 0.1, 76.6, 29.2),
    (2019, "ISSCC", "20.6", "SDCT", "0.028", 0.16, 64.9, 7.33),
    (2019, "ISSCC", "20.7", "SDCT", "0.028", 0.2, 72.6, 16.3),
    (2019, "ISSCC", "3.1", "Pipe, TI", "0.016", 3.2, 61.7, 61.3),
    (2019, "ISSCC", "3.2", "Pipe, SAR", "0.028", 1.0, 60.0, 7.6),
    (2019, "ISSCC", "3.3", "Pipe, SAR, TI", "0.028", 5.0, 58.5, 158.6),
    (2019, "ISSCC", "3.6", "Pipe", "0.016", 0.6, 60.2, 6.0),
    (2019, "VLSI", "7.3", "SAR, Pipe", "0.04", 0.2, 62.1, 3.9),
    (2019, "VLSI", "7.5", "SAR, TI", "0.028", 5.0, 48.5, 29.0),
    (2020, "ISSCC", "16.1", "Pipe, TI", "0.016", 18.0, 48.0, 1300.0),
    (2020, "ISSCC", "16.2", "TI, TDC", "0.065", 10.0, 40.1, 50.8),
    (2020, "ISSCC", "16.4", "Pipe, SAR", "0.028", 0.1, 71.7, 0.7),
    (2020, "ISSCC", "16.6", "Pipe, VCO", "0.016", 1.6, 58.0, 280.0),
    (2020, "ISSCC", "5.1", "VCO", "0.028", 0.2, 39.3, 1.38),
    (2020, "VLSI", "CD1.1", "Pipe", "0.016", 1.0, 59.5, 10.9),
    (2020, "VLSI", "CD1.2", "SAR", "0.04", 0.1, 56.3, 1.4),
    (2020, "VLSI", "CD1.3", "SAR, Flash", "0.028", 1.0, 45.47, 2.55),
    (2020, "VLSI", "CD2.3", "SDCT", "0.028", 0.1, 74.4, 10.4),
    (2020, "VLSI", "CD2.4", "SAR, Pipe", "0.028", 0.25, 63.6, 2.3),
    (2021, "ISSCC", "10.3", "SDCT, BP", "0.028", 0.2, 67.5, 13.4),
    (2021, "ISSCC", "10.5", "Pipe, SDCT", "0.007", 0.6, 55.3, 13.0),
    (2021, "ISSCC", "27.5", "SAR, TI", "0.022", 0.16, 66.32, 2.56),
    (2021, "VLSI", "C15-1", "SAR, Pipe", "0.016", 0.5, 62.3, 2.8),
    (2021, "VLSI", "C15-2", "Time-Based, TI", "0.065", 20.0, 38.8, 129.3),
    (2021, "VLSI", "C15-3", "SAR, TI", "0.016", 8.0, 42.4, 26.0),
    (2021, "VLSI", "C15-4", "SAR, Pipe", "0.028", 1.0, 59.28, 19.2),
    (2022, "ISSCC", "10.1", "Time-based", "0.014", 5.0, 40.8, 7.4),
    (2022, "ISSCC", "10.2", "Pipe, SAR", "0.028", 0.13, 72.5, 0.82),
    (2022, "ISSCC", "10.3", "Pipe, SAR", "0.028", 0.2, 66.7, 1.3),
    (2022, "ISSCC", "10.4", "Pipe, SAR", "0.022", 0.26, 60.5, 0.97),
    (2022, "ISSCC", "25.4", "SDCT", "0.04", 0.72, 65.0, 158.0),
    (2022, "VLSI", "C10-3", "SAR", "0.008", 0.25, 62.0, 0.56),
    (2022, "VLSI", "C10-4", "Subranging", "0.028", 2.5, 44.8, 3.5),
    (2022, "VLSI", "C19-2", "SAR, TI", "0.022", 3.8, 38.0, 6.0),
    (2022, "VLSI", "C19-3", "SAR, Pipe, TI", "0.005", 10.0, 48.0, 625.0),
    (2022, "VLSI", "C19-4", "TI", "0.028", 1.4, 48.0, 9.86),
    (2023, "ISSCC", "10.1", "Pipe, TI", "0.007", 1.8, 60.16, 7.55),
    (2023, "ISSCC", "10.2", "Pipe, Time-based", "0.028", 2.6, 51.4, 13.9),
    (2023, "ISSCC", "10.3", "Pipe, Time-based", "0.028", 2.0, 60.4, 27.0),
    (2023, "ISSCC", "10.7", "SAR", "0.028", 0.2, 70.15, 4.52),
    (2023, "ISSCC", "17.1", "SAR, TI", "0.028", 2.8, 51.8, 18.05),
    (2023, "ISSCC", "17.2", "Time-based", "0.005", 1.25, 39.3, 1.9),
    (2023, "ISSCC", "17.4", "SAR, TI", "0.007", 24.0, 46.5, 750.0),
    (2023, "ISSCC", "17.5", "Pipe, SAR, TI", "0.028", 1.0, 62.5, 10.6),
    (2023, "ISSCC", "17.6", "SAR, TI", "0.028", 4.5, 38.33, 6.56),
    (2023, "ISSCC", "17.7", "Subranging", "0.028", 2.7, 45.9, 3.0),
    (2023, "VLSI", "C14-1", "SAR, TI", "0.028", 2.0, 57.3, 118.6),
    (2023, "VLSI", "C14-3", "SAR, Pipe", "0.028", 0.15, 67.9, 1.72),
    (2023, "VLSI", "C4-1", "Pipe, CT", "0.016", 2.0, 60.0, 240.0),
    (2023, "VLSI", "C4-3", "SDCT", "0.028", 0.24, 72.8, 115.0),
    (2023, "VLSI", "C4-4", "SDCT", "0.022", 0.44, 62.0, 22.0),
    (2024, "ISSCC", "18.1", "SAR, TI", "0.016", 105.0, 39.2, 698.0),
    (2024, "ISSCC", "22.1", "Pipe, TI", "0.028", 12.0, 54.1, 179.8),
    (2024, "ISSCC", "22.2", "Pipe, CT", "0.016", 1.4, 71.2, 703.0),
    (2024, "ISSCC", "22.4", "SAR, TI", "0.028", 4.8, 44.3, 7.7),
    (2024, "ISSCC", "9.1", "SAR, TI", "0.022", 0.2, 70.7, 2.0),
    (2024, "ISSCC", "9.2", "Pipe, SAR", "0.008", 0.4, 62.8, 2.08),
    (2024, "ISSCC", "9.3", "Pipe, SAR, TI", "0.028", 0.4, 71.2, 12.5),
    (2024, "VLSI", "C18.4", "SAR, Pipe", "0.038", 0.475, 65.9, 6.29),
    (2024, "VLSI", "C18.5", "Pipe", "0.016", 1.0, 67.45, 17.8),
    (2024, "VLSI", "C24.2", "Time-based, Pipe, SAR, TI", "0.004", 16.0, 44.48, 94.2),
    (2024, "VLSI", "C24.4", "SAR, TI", "0.005", 10.0, 50.2, 386.0),
    (2024, "VLSI", "C24.5", "Pipe, TI", "0.016", 10.0, 51.0, 350.0),
    (2024, "VLSI", "C8.5", "SDCT", "0.065", 0.168, 62.0, 53.0),
    (2025, "ISSCC", "18.7", "Pipe, SAR", "0.028", 0.16, 70.1, 4.87),
    (2025, "ISSCC", "18.8", "SAR", "0.04", 0.8, 40.74, 2.89),
    (2025, "ISSCC", "24.1", "Pipe", "0.028", 3.0, 58.8, 32.5),
    (2025, "ISSCC", "24.2", "Pipe", "0.028", 1.0, 68.2, 15.3),
    (2025, "ISSCC", "24.3", "Time-based, TI", "0.028", 2.2, 45.8, 6.9),
    (2025, "ISSCC", "24.4", "Time-based", "0.028", 3.0, 49.3, 22.9),
    (2025, "ISSCC", "24.5", "Pipe, SAR, TI", "0.016", 72.0, 41.9, 393.3),
    (2025, "ISSCC", "24.6", "Pipe, SAR, TI", "0.004", 16.0, 47.0, 570.0),
    (2025, "ISSCC", "24.7", "Pipe, TI", "0.028", 10.0, 41.7, 21.9),
    (2025, "ISSCC", "24.8", "SAR, TI", "0.016", 12.0, 48.1, 160.0),
    (2025, "VLSI", "C5-1", "Pipe, SAR, TI", "0.028", 0.35, 59.7, 3.38),
    (2025, "VLSI", "C5-2", "SDCT", "0.028", 0.2, 74.5, 49.5),
    (2025, "VLSI", "C5-3", "SAR", "0.065", 0.1, 48.7, 0.599),
    (2025, "VLSI", "C8-1", "Subranging", "0.028", 0.56, 72.14, 9.76),
    (2025, "VLSI", "C8-3", "Pipe", "0.028", 3.0, 62.3, 125.4),
    (2026, "ISSCC", "11.1", "Pipe, SAR, Time-based", "0.028", 0.4, 67.0, 4.59),
    (2026, "ISSCC", "11.2", "Pipe, SAR", "0.028", 0.54, 69.5, 155.0),
    (2026, "ISSCC", "11.3", "Pipe, SAR", "0.028", 0.5, 63.4, 2.69),
    (2026, "ISSCC", "11.4", "Pipe, SAR", "0.028", 0.5, 67.5, 10.8),
    (2026, "ISSCC", "11.7", "Pipe, TI", "0.005", 12.0, 52.5, 1300.0),
    (2026, "VLSI", "C17.1", "Pipe, TI, Time-based", "0.028", 4.0, 58.5, 43.85),
    (2026, "VLSI", "C17.2", "Pipe, Time-based", "0.028", 0.8, 67.0, 25.8),
    (2026, "VLSI", "C17.3", "SAR, TI", "0.014", 4.8, 41.5, 5.1),
    (2026, "VLSI", "C17.4", "Pipe, SAR", "0.028", 0.5, 70.12, 7.6),
    (2026, "VLSI", "C17.5", "Pipe, SAR", "0.028", 2.5, 60.8, 19.4),
    (2026, "VLSI", "C28.1", "SDCT, SAR", "0.022", 0.24, 70.2, 21.9),
    (2026, "VLSI", "C28.4", "Pipe, CT", "0.028", 0.2, 68.8, 30.0),
)

SURVEY = dict(rows=764, usable=763, years=(1997, 2026))
BITS_KEPT, RATE_KEPT = 6, 0.1    # the table holds every part at or above these
RANK = 5                         # the survey's own envelope is drawn through five points
PUBLISHED = (                    # from the papers themselves: name, row, GS/s, dB, mW
    ("Kull et al., JSSC 2013", (2013, "ISSCC", "26.4"), 1.2, 39.3, 3.1),
    ("Verbruggen et al., JSSC 2010", (2010, "ISSCC", "16.3"), 2.2, 31.1, 2.6),
)
# what the reading from the slides had, for the record of what the spreadsheet moved
SLIDES = dict(matched=546, markers=(596, 613, 569), in_sheet=541, to_2021=618, not_given=77,
              below_axis=43, near=34, quartile_fj=40.3, sheet_near=43, sheet_quartile_fj=85.8)


def sndr_db(bits):
    """The SNDR an ideal converter of that many bits has."""
    return 6.02 * bits + 1.76


def enob(sndr):
    return (sndr - 1.76) / 6.02


def sample_j(row):
    """Joules a sample, P / fs: what the part spends each time it converts."""
    return row[P] * 1e-3 / (row[FS] * 1e9)


def walden_j(row):
    """Joules a conversion step, P / (2^ENOB fs)."""
    return sample_j(row) / 2 ** enob(row[SNDR])


def cite(row):
    return f"{row[VENUE]} {row[YEAR]} {row[ID]}"


def able(bits, fs_hz, table=TABLE, year=None):
    """The parts that can do a column's job: at least `bits` effective bits at
    `fs_hz` or faster, published by `year` if one is given.  Cheapest sample first."""
    return sorted((r for r in table
                   if r[SNDR] >= sndr_db(bits) - 1e-6 and r[FS] * 1e9 >= fs_hz * (1 - 1e-9)
                   and (year is None or r[YEAR] <= year)), key=sample_j)


def price_j(bits, fs_hz, table=TABLE):
    """(best, fifth-best) joules a sample among the parts that can do the job.
    The last there is, where fewer than five can."""
    rows = able(bits, fs_hz, table)
    return sample_j(rows[0]), sample_j(rows[min(RANK, len(rows)) - 1])


def as_published(bits, fs_hz, table=TABLE):
    """The part that does the job for the least power, run as it was measured."""
    return min(able(bits, fs_hz, table), key=lambda r: r[P])


def staircase(bits, table=TABLE):
    """[(GS/s, row)]: the cheapest part for every rate up to that one, in order.
    Its rate is where it stops being able to do the job and the next takes over."""
    out, rate = [], RATE_KEPT * 1e9
    while True:
        rows = able(bits, rate, table)
        if not rows:
            return out
        out.append((rows[0][FS], rows[0]))
        rate = rows[0][FS] * 1e9 * (1 + 1e-6)


# ---- re-reading the spreadsheet -------------------------------------------------
_T = '{urn:oasis:names:tc:opendocument:xmlns:table:1.0}'
_O = '{urn:oasis:names:tc:opendocument:xmlns:office:1.0}'
_X = '{urn:oasis:names:tc:opendocument:xmlns:text:1.0}'


def _sheet(tab, width=40):
    """A sheet's rows, a row a list of floats, strings and Nones."""
    rows = []
    for row in tab.iter(_T + 'table-row'):
        cells = []
        for cell in row:
            if cell.tag not in (_T + 'table-cell', _T + 'covered-table-cell'):
                continue
            kind = cell.get(_O + 'value-type')
            if kind in ('float', 'percentage', 'currency'):
                v = float(cell.get(_O + 'value'))
            elif kind is None:
                v = None
            else:
                v = ' '.join(''.join(p.itertext()) for p in cell.iter(_X + 'p')).strip() or None
            rep = int(cell.get(_T + 'number-columns-repeated', '1'))
            cells.extend([v] * min(rep, width - len(cells)))
        if any(c is not None for c in cells):
            rows.append(cells + [None] * (width - len(cells)))
    return rows


def _text(v):
    return '%g' % v if isinstance(v, float) else str(v)


def read_ods(path):
    """Every row of the survey's two sheets that gives a power, a rate and an
    SNDR, unrounded, in TABLE's order; and how many rows there were in all."""
    root = ET.fromstring(zipfile.ZipFile(path).read('content.xml'))
    out, seen = [], 0
    for tab in root.iter(_T + 'table'):
        venue = tab.get(_T + 'name')
        if venue not in ('ISSCC', 'VLSI'):
            continue
        rows = _sheet(tab)
        col = {name: i for i, name in enumerate(rows[0]) if name}
        for r in rows[1:]:
            seen += 1
            p, fs, sndr = r[col['P [W]']], r[col['fsnyq [Hz]']], r[col['SNDR_plot [dB]']]
            if not all(isinstance(v, float) for v in (p, fs, sndr)):
                continue
            out.append((int(float(r[col['YEAR']])), venue, _text(r[col['ID']]), _text(r[col['ARCHITECTURE']]),
                        _text(r[col['TECHNOLOGY']]), fs / 1e9, sndr, p * 1e3))
    return out, seen


def kept(rows):
    """The part of the survey this file keeps, rounded as TABLE is."""
    out = [r[:FS] + (float('%.4g' % r[FS]), round(r[SNDR], 2), float('%.4g' % r[P])) for r in rows]
    return tuple(sorted(r for r in out if r[SNDR] >= sndr_db(BITS_KEPT) - 1e-6 and r[FS] >= RATE_KEPT * (1 - 1e-9)))


# ---- re-reading the slides ----------------------------------------------------
NUM = r'(-?\d+(?:\.\d+)?)'


def _inflate(raw):
    try:
        return zlib.decompress(raw)
    except zlib.error:
        return zlib.decompressobj().decompress(raw)


def _objects(data):
    """Every object in the file: number -> (dictionary bytes, decoded stream or None)."""
    objs = {}
    for m in re.finditer(rb'(\d+)\s+(\d+)\s+obj\b(.*?)\bendobj', data, re.S):
        body = m.group(3)
        s = re.search(rb'stream\r?\n', body)
        if not s:
            objs[int(m.group(1))] = (body, None)
            continue
        head, raw = body[:s.start()], body[s.end():]
        end = raw.rfind(b'endstream')
        raw = (raw[:end] if end >= 0 else raw).rstrip(b'\r\n')
        objs[int(m.group(1))] = (head, _inflate(raw) if b'/FlateDecode' in head else raw)
    for head, stream in list(objs.values()):
        if stream is None or b'/ObjStm' not in head:
            continue
        n = int(re.search(rb'/N\s+(\d+)', head).group(1))
        first = int(re.search(rb'/First\s+(\d+)', head).group(1))
        nums = stream[:first].split()
        pairs = [(int(nums[2 * i]), int(nums[2 * i + 1])) for i in range(n)]
        for i, (num, off) in enumerate(pairs):
            end = pairs[i + 1][1] if i + 1 < n else len(stream) - first
            objs.setdefault(num, (stream[first + off:first + end], None))
    return objs


def _value(objs, head, key):
    """/key's value in a dictionary, following one reference."""
    m = re.search(rb'/' + key + rb'\b\s*', head)
    if not m:
        return b''
    rest = head[m.end():]
    r = re.match(rb'(\d+)\s+\d+\s+R', rest)
    return objs[int(r.group(1))][0] if r else rest


def _pages(objs):
    def walk(num, out, seen):
        if num in seen or num not in objs:
            return
        seen.add(num)
        head = objs[num][0]
        if re.search(rb'/Type\s*/Pages\b', head):
            kids = _value(objs, head, b'Kids')
            for k in re.findall(rb'(\d+)\s+\d+\s+R', kids[:kids.index(b']') + 1]):
                walk(int(k), out, seen)
        elif re.search(rb'/Type\s*/Page\b', head):
            out.append(num)

    out, seen = [], set()
    for head, _ in objs.values():
        if re.search(rb'/Type\s*/Catalog\b', head):
            root = re.search(rb'/Pages\s+(\d+)\s+\d+\s+R', head)
            walk(int(root.group(1)), out, seen)
    return out


def _plot(objs, pages, number):
    """A slide's plot: its grid lines and its markers' centres, in drawing units."""
    head = objs[pages[number - 1]][0]
    c = re.search(rb'/Contents\s+(\d+)\s+\d+\s+R', head)
    name = re.search(rb'/(Meta\d+)\s+Do', objs[int(c.group(1))][1]).group(1)
    xo = _value(objs, _value(objs, head, b'Resources'), b'XObject')
    form = re.search(rb'/' + name + rb'\s+(\d+)\s+\d+\s+R', xo)
    s = objs[int(form.group(1))][1].decode('latin-1')
    area = re.search(NUM + ' ' + NUM + ' ' + NUM + ' ' + NUM + r' re\s*\nf\*', s)
    w, h = float(area.group(3)), float(area.group(4))
    vx, hy = set(), set()
    for m in re.finditer(NUM + ' ' + NUM + r' m\s*\n' + NUM + ' ' + NUM + r' l\s*\nS', s):
        ax, ay, bx, by = map(float, m.groups())
        if ax == bx and abs(abs(by - ay) - h) < 1.0:
            vx.add(ax)
        elif ay == by and abs(abs(bx - ax) - w) < 1.0:
            hy.add(ay)
    pts = []
    for m in re.finditer(NUM + ' ' + NUM + ' ' + NUM + ' ' + NUM + r' re\s*\nf\*', s[s.index('0 0 1 rg'):]):
        x, y, mw, mh = map(float, m.groups())
        if 3.0 < mw <= 13.0 and 3.0 < mh <= 13.0:
            pts.append((x + mw / 2, y + mh / 2))
    return sorted(vx), sorted(hy), pts


def _axis(positions, values):
    """Drawing units to data, least squares through the grid lines; and the worst miss."""
    n = len(positions)
    assert n == len(values), (n, len(values))
    mp, mv = sum(positions) / n, sum(values) / n
    k = sum((p - mp) * (v - mv) for p, v in zip(positions, values)) / sum((p - mp) ** 2 for p in positions)
    return (lambda p: mv + k * (p - mp)), max(abs(mv + k * (p - mp) - v) for p, v in zip(positions, values))


def read_slides(path):
    """Every converter three plots of the short course agree on, as (GS/s, dB,
    mW, FoM_S); the three marker counts; and the worst miss of the grid fit.

    Slide 45 plots rate against SNDR, slide 59 SNDR against energy a
    conversion, slide 64 rate against FoM_S = SNDR + 10 log10((fs / 2) / P).
    Slide 59's point gives a FoM_S, and slide 64 has to show it at slide 45's
    rate."""
    objs = _objects(open(path, 'rb').read())
    pages = _pages(objs)
    (avx, ahy, a), (bvx, bhy, b), (cvx, chy, c) = (_plot(objs, pages, n) for n in (45, 59, 64))
    ax, e1 = _axis(avx, list(range(4, 12)))                 # log10 of 10^4 .. 10^11 samples a second
    ay, e2 = _axis(ahy, [20, 40, 60, 80, 100])              # dB
    bx, e3 = _axis(bvx, list(range(10, 111, 10)))           # dB
    by, e4 = _axis(bhy, [-14, -12, -10, -8, -6])            # log10 of joules
    cx, e5 = _axis(cvx, list(range(3, 12)))
    cy, e6 = _axis(chy, list(range(130, 191, 10)))          # dB
    A = [(ax(x), ay(y)) for x, y in a]
    B = [(bx(x), by(y)) for x, y in b]
    C = [(cx(x), cy(y)) for x, y in c]
    rows, used_b, used_c = [], set(), set()
    for lf, sndr in A:
        best = None
        for i, (s2, le) in enumerate(B):
            if i in used_b or abs(s2 - sndr) > 0.06:
                continue
            foms = s2 - 10 * (math.log10(2.0) + le)
            for j, (lf2, f2) in enumerate(C):
                if j in used_c or abs(lf2 - lf) > 0.004 or abs(f2 - foms) > 0.12:
                    continue
                err = abs(s2 - sndr) / 0.06 + abs(lf2 - lf) / 0.004 + abs(f2 - foms) / 0.12
                if best is None or err < best[0]:
                    best = (err, i, j, le, f2)
        if best:
            used_b.add(best[1])
            used_c.add(best[2])
            rows.append((10 ** (lf - 9), sndr, 10 ** (best[3] + lf + 3), best[4]))
    return rows, (len(A), len(B), len(C)), max(e1, e2, e3, e4, e5, e6)


def _same(slide, row):
    """How far a converter read from the slides is from a row, or None if it is not it."""
    d = (abs(math.log10(row[FS] / slide[0])) / 0.004, abs(row[SNDR] - slide[1]) / 0.06,
         abs(math.log10(row[P] / slide[2])) / 0.004)
    return sum(d) if max(d) < 1.0 else None


def pair(slides, rows):
    """Each slide-read converter to a row of its own: {slide index: row index}."""
    used, out = set(), {}
    for i, s in enumerate(slides):
        c = [(d, j) for j, r in enumerate(rows) if j not in used for d in (_same(s, r),) if d is not None]
        if c:
            out[i] = min(c)[1]
            used.add(out[i])
    return out


def foms_db(row):
    return row[SNDR] + 10 * math.log10(row[FS] * 1e9 / 2 / (row[P] * 1e-3))


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print("The ADC survey, for what a photonic tile's column asks of its converter.")
    print(f"{SURVEY['usable']} operating points from ISSCC and the VLSI Symposium, {SURVEY['years'][0]} to {SURVEY['years'][1]};"
          f" the {len(TABLE)} of at")
    print(f"least {BITS_KEPT} effective bits at {RATE_KEPT} GS/s and above are kept here.")

    section("1. The parts that can do the job: B effective bits or more, at the rate or faster")
    print(f"  {'bits':>4}{'rate':>10}{'parts':>7}{'best':>10}{'fifth':>11}{'tenth':>11}   cheapest run as it was measured")
    for bits in (6, 7, 8):
        for fs in (0.1e9, 1e9, 10e9):
            rows = able(bits, fs)
            j = [sample_j(r) * 1e12 for r in rows]
            a = as_published(bits, fs)
            tenth = f"{j[9]:>8.2f} pJ" if len(j) > 9 else f"{'-':>11}"
            print(f"  {bits:>4}{fs / 1e9:>5.1f} GS/s{len(rows):>7}{j[0]:>7.2f} pJ{j[min(RANK, len(j)) - 1]:>8.2f} pJ{tenth}"
                  f"   {a[P]:.2f} mW, {cite(a)} at {a[FS]:.2f} GS/s")
    print("  Energy a sample, P / fs.  A part faster than the rate is priced at the")
    print("  rate, which ASSUMES its power follows its clock down.  The last column")
    print("  assumes nothing.")

    section("2. Version 1's converter, 7 bits at 1 GS/s: the cheapest that can")
    print(f"  {'':<18}{'':<24}{'um':>6}{'GS/s':>7}{'SNDR':>8}{'mW':>7}{'pJ a sample':>13}{'fJ a step':>11}")
    for r in able(7, 1e9)[:8]:
        print(f"  {cite(r):<18}{r[ARCH][:22]:<24}{r[NODE]:>6}{r[FS]:>7.2f}{r[SNDR]:>6.1f} dB{r[P]:>7.2f}"
              f"{sample_j(r) * 1e12:>13.2f}{walden_j(r) * 1e15:>11.1f}")
    b6, b7, b8 = (price_j(b, 1e9) for b in (6, 7, 8))
    print(f"  The best 6-bit part is {b6[0] * 1e12:.2f} pJ and the best 7-bit one {b7[0] * 1e12:.2f}: at the head of the")
    print(f"  field the seventh bit is {(b7[0] / b6[0] - 1) * 100:.0f}%, because the best converters at this rate are")
    print(f"  8-bit designs that reach 7.  At the fifth-best it is {b7[1] / b6[1]:.1f} times, and the eighth")
    print(f"  bit is {b8[0] / b7[0]:.1f} times at the head.")

    section("3. By rate: where the cheapest part stops being able to do it")
    for bits in (6, 7):
        print(f"  {bits} bits")
        for rate, r in staircase(bits):
            print(f"    up to {rate:>6.2f} GS/s  {sample_j(r) * 1e12:>6.2f} pJ a sample   {cite(r)}, {r[SNDR]:.1f} dB, {r[P]:.2f} mW")
    j1, j10 = price_j(7, 1e9), price_j(7, 10e9)
    print(f"  At 7 bits ten times the rate, 1 to 10 GS/s, is {j10[0] / j1[0]:.1f} times the energy a sample")
    print(f"  at the best and {j10[1] / j1[1]:.0f} at the fifth: {10 * j10[0] / j1[0]:.0f} and {10 * j10[1] / j1[1]:.0f} times the power.")
    print(f"  Only {len(able(7, 10e9))} published converters do 7 bits at 10 GS/s at all.")

    section("4. By year: the best 7-bit converter at 1 GS/s or more, as it stood")
    last = None
    for year in range(SURVEY['years'][0], SURVEY['years'][1] + 1):
        rows = able(7, 1e9, year=year)
        if rows and rows[0] != last:
            last = rows[0]
            print(f"    {year}  {len(rows):>3} able  {sample_j(last) * 1e12:>7.2f} pJ   {cite(last)}, {last[FS]:.2f} GS/s, {last[P]:.2f} mW")

    checks()
    ods = [a for a in sys.argv[1:] if not a.lower().endswith('.pdf')]
    pdf = [a for a in sys.argv[1:] if a.lower().endswith('.pdf')]
    sheet = slides = None
    if ods:
        sheet = check_sheet(ods[0])
    if pdf:
        slides = check_slides(pdf[0])
    if sheet and slides:
        check_both(sheet, slides)


def checks():
    """Every claim above, as an assert."""
    assert len(TABLE) == len(set(TABLE)) == 226
    assert list(TABLE) == sorted(TABLE)
    assert all(r[SNDR] >= sndr_db(BITS_KEPT) - 1e-6 and r[FS] >= RATE_KEPT for r in TABLE)
    assert SURVEY['years'][0] <= min(r[YEAR] for r in TABLE) and max(r[YEAR] for r in TABLE) == SURVEY['years'][1]

    # 1. Version 1's converter, 7 bits at 1 GS/s: 70 published parts can do it.
    #    The best is 1.11 pJ a sample and the fifth-best 3.14.  The cheapest run
    #    as it was measured is a part at exactly 1.00 GS/s, for 2.55 mW, so the
    #    converter exists and is not a projection.
    rows = able(7, 1e9)
    lo, hi = price_j(7, 1e9)
    assert len(rows) == 70
    assert rows[0][:ID + 1] == (2023, "ISSCC", "17.7") and abs(lo * 1e12 - 1.11) < 0.005
    assert abs(hi * 1e12 - 3.14) < 0.005 and hi == sample_j(rows[RANK - 1])
    a = as_published(7, 1e9)
    assert a[:ID + 1] == (2020, "VLSI", "CD1.3") and (a[FS], a[P]) == (1.0, 2.55)
    assert lo < sample_j(a) < hi

    # 2. Version 0's, 6 bits: 89 parts, 1.06 and 1.48 pJ.  The seventh bit is 5%
    #    at the head of the field and 2.1 times at the fifth-best; the eighth is
    #    3.8 times at the head.
    lo6, hi6 = price_j(6, 1e9)
    lo8, _ = price_j(8, 1e9)
    assert len(able(6, 1e9)) == 89
    assert abs(lo6 * 1e12 - 1.06) < 0.005 and abs(hi6 * 1e12 - 1.48) < 0.005
    assert 1.04 < lo / lo6 < 1.06 and 2.1 < hi / hi6 < 2.2 and 3.7 < lo8 / lo < 3.8
    # because three of the eight cheapest 6-bit parts are 7-bit parts
    assert sum(1 for r in able(6, 1e9)[:8] if r[SNDR] >= sndr_db(7)) == 3

    # 3. A dearer job never has a cheaper best part: the parts that can do it
    #    are a subset.  So the energy a sample never falls as the rate or the
    #    resolution rises.
    grid = (0.1e9, 0.3e9, 1e9, 3e9, 10e9)
    for bits in (6, 7, 8):
        j = [price_j(bits, fs)[0] for fs in grid]
        assert j == sorted(j), (bits, j)
    for fs in grid:
        j = [price_j(bits, fs)[0] for bits in (6, 7, 8)]
        assert j == sorted(j), (fs, j)

    # 4. At 7 bits the best part is the same one from 0.1 to 2.7 GS/s, so the
    #    ADCs' power is linear in the shot rate that far.  By 10 GS/s the
    #    energy a sample is 5.3 times as much, which is 53 times the power for
    #    ten times the rate; at the fifth-best it is over a hundred.  Eleven
    #    converters do it at all.
    stairs = staircase(7)
    assert stairs[0][1] == rows[0] and stairs[0][0] == 2.7
    assert [round(sample_j(r) * 1e12, 2) for _, r in stairs] == [1.11, 1.6, 4.54, 5.89, 31.25]
    j10 = price_j(7, 10e9)
    assert len(able(7, 10e9)) == 11
    assert abs(j10[0] / lo - 5.3) < 0.05 and 10 * j10[1] / hi > 100
    # and at 6 bits it is 2.1 times, 21 times the power
    assert abs(price_j(6, 10e9)[0] / lo6 - 2.06) < 0.01

    # 5. The slides stopped at 2021.  The best then was the part at exactly
    #    1 GS/s, 2.55 pJ; three cheaper have been published since, and the best
    #    is 2.3 times better.
    old = able(7, 1e9, year=2021)
    assert len(old) == 41 and old[0] == a
    assert [r[YEAR] for r in rows[:4]] == [2023, 2022, 2024, 2020]
    assert 2.25 < sample_j(old[0]) / lo < 2.35

    # 6. Where a paper's title states its own figure of merit, the row gives it:
    #    "9.8-fJ/conv.-step" and "10.9fJ/conv.-step".
    by = {r[:ID + 1]: r for r in TABLE}
    assert abs(walden_j(by[(2022, "VLSI", "C10-4")]) * 1e15 - 9.8) < 0.15
    assert abs(walden_j(by[(2026, "VLSI", "C17.3")]) * 1e15 - 10.9) < 0.05

    # 7. The converter pta_power.py first held from its own paper is here, at
    #    the paper's rate and SNDR and within 2% of its power.
    name, key, fs, sndr, p = PUBLISHED[0]
    r = by[key]
    assert (r[FS], r[SNDR]) == (fs, sndr) and abs(r[P] / p - 1.0) < 0.02, (name, r)

    print("\nAll checks pass.")


def check_sheet(path):
    """Hold the table to the spreadsheet, and what the slides' reading had to it."""
    rows, seen = read_ods(path)
    assert (seen, len(rows)) == (SURVEY['rows'], SURVEY['usable']), (seen, len(rows))
    assert (min(r[YEAR] for r in rows), max(r[YEAR] for r in rows)) == SURVEY['years']
    got = kept(rows)
    assert got == TABLE, sorted(set(got) ^ set(TABLE))[:6]
    # the second paper pta_power.py held is below this table's 6 bits, and is
    # in the spreadsheet at its own figures
    name, key, fs, sndr, p = PUBLISHED[1]
    hit = [r for r in rows if r[:ID + 1] == key]
    assert len(hit) == 1 and abs(hit[0][FS] - fs) < 1e-9 and abs(hit[0][SNDR] - sndr) < 1e-9 and abs(hit[0][P] / p - 1) < 0.02
    # the first reading's statistic -- within 2.5 of 1 GS/s, 30 to 50 dB, to
    # 2021 -- on the whole record: 43 parts and not 34, the same best, and a
    # quarter mark of 86 fJ a step and not 40
    near = sorted((r for r in rows if r[YEAR] <= 2021 and 0.4 <= r[FS] <= 2.5 and 30.0 <= r[SNDR] <= 50.0), key=walden_j)
    assert len(near) == SLIDES['sheet_near'], len(near)
    assert abs(walden_j(near[0]) * 1e15 - 16.6) < 0.05
    assert abs(walden_j(near[(len(near) - 1) // 4]) * 1e15 - SLIDES['sheet_quartile_fj']) < 0.05
    print(f"\nRe-read from {path}: {seen} rows, {len(rows)} with a power, a rate and an SNDR.")
    print("The table is what the spreadsheet holds.")
    return rows


def check_slides(path):
    """Hold the slides' plots to the table."""
    rows, counts, fit = read_slides(path)
    assert (len(rows), counts) == (SLIDES['matched'], SLIDES['markers']), (len(rows), counts)
    assert fit < 0.002, fit
    # every converter the plots give that is clear of the table's edges is in
    # the table, published by 2021
    old = [r for r in TABLE if r[YEAR] <= 2021]
    inside = [s for s in rows if s[1] >= sndr_db(BITS_KEPT) + 0.1 and s[0] >= RATE_KEPT * 1.01]
    found = pair(inside, old)
    assert len(found) == len(inside), (len(found), len(inside))
    print(f"\nRe-read from {path}: {counts[0]}, {counts[1]} and {counts[2]} markers,"
          f" {len(rows)} matched, grid fit {fit:.4f}.")
    print(f"All {len(inside)} of them at {BITS_KEPT} bits and {RATE_KEPT} GS/s are in the table.")
    return rows


def check_both(sheet, slides):
    """Hold the slides' reading to the whole spreadsheet, as it stood in 2021."""
    old = [r for r in sheet if r[YEAR] <= 2021]
    found = pair(slides, old)
    assert (len(old), len(found)) == (SLIDES['to_2021'], SLIDES['in_sheet']), (len(old), len(found))
    worst_db = max(abs(old[j][SNDR] - slides[i][1]) for i, j in found.items())
    worst_p = max(abs(old[j][P] / slides[i][2] - 1) for i, j in found.items())
    assert worst_db < 0.04 and worst_p < 0.001, (worst_db, worst_p)
    # the five that are not converters: none is at the table's bits and rate
    false = [s for i, s in enumerate(slides) if i not in found]
    assert len(false) == SLIDES['matched'] - SLIDES['in_sheet']
    assert all(s[0] < RATE_KEPT for s in false)
    # and what the plots never gave: 77 rows, 43 of them under slide 64's axis
    given = set(found.values())
    missing = [r for j, r in enumerate(old) if j not in given]
    assert len(missing) == SLIDES['not_given']
    assert sum(1 for r in missing if foms_db(r) < 130.0) == SLIDES['below_axis']
    assert min(foms_db(old[j]) for j in given) > 130.0
    near = [r for r in old if 0.4 <= r[FS] <= 2.5 and 30.0 <= r[SNDR] <= 50.0]
    lost = [r for r in near if r in missing]
    assert len(near) - len(lost) == SLIDES['near'] and all(walden_j(r) > 3000e-15 for r in lost)
    print(f"The slides against the spreadsheet to 2021: {len(found)} of {len(slides)} matched converters are")
    print(f"rows of it, to {worst_db:.2f} dB and {worst_p * 100:.2f}% in power.  The other {len(false)} are chance")
    print(f"agreements of three plots.  {len(missing)} of its {len(old)} rows the plots never gave.")


if __name__ == "__main__":
    main()
