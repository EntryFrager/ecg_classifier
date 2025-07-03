mkdir ./data
cd ./data
wget -r -N --no-parent --accept .csv,.dat,.hea,.py --reject .html,.tmp -c https://physionet.org/files/ptb-xl/1.0.1/
cd ../
