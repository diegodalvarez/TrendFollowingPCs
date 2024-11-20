# Trend Following Principal Component Analysis
Trend Following Strategies for Principal Components

## Overview
This repo goal is to apply trend following techniques for various Principal Component techniques. In this case trend following strategies using EWMACs and EWMAs for the yield curve did not prove successful. Although decomposing Treasury futures (which can be considered as at term structure) and then taking the difference between yield PCs and Futures PCs and then applying Trend Following filters (EWAMCs and EWMAs) on top did prove successful.

Another common approach in trend following is applying a Kalman Filter which give both a smooth component and a residual. This is useful because the smooth component can be used for applying a trend follower (which failed), but trading the mean reversion of the residuals by shorting z-scores did work. This can be applied to the principal components of the yield curve to trade Treasury Futures which proved useful.
