mkdir ./dataset
cd ./dataset
wget -r -N --no-parent --accept .csv,.dat,.hea --reject .html,.tmp,.py -c https://physionet.org/files/ptb-xl/1.0.1/
cd ../