function fib(){
    if [ $1 -le 0 ]; then
        echo 0
    elif [ $1 -eq 1 ]; then
        echo 1
    else
        echo $[`fib $[$1-2]` + `fib $[$1 - 1]` ]
    fi

}

fib $1
