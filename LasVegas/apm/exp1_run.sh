i=1
end=10

#Iterate the loop until a less than equal to $end
while [ $i -le $end ]
do
    # Print the values
    OUT_FILE_NAME="output/"$2"/output_"$2"-Bits_File-"$i".txt"

    echo "Processing try-cnt :: "$i"   writing to file :: "$OUT_FILE_NAME

    time mpirun -n $3 $1 > $OUT_FILE_NAME
    #mpirun  -machinefile $PBS_NODEFILE  -np $3 ./$1 > $OUT_FILE_NAME

    # increment the value
    i=`expr $i + 1`
done

# $1 : executable file name
# $2 : folder name inside the output folder
# $3 : number of processors
