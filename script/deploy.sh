#!/bin/bash
script_path="$( cd "$(dirname "$0")" ; pwd -P )"
app_path="${script_path}/../src"

. ${script_path}/func_util.sh

function build_common()
{
	echo "build common lib..."
    bash ${script_path}/build_ezdvpp.sh ${remote_host}
    if [ $? -ne 0 ];then
        echo "ERROR: Failed to deploy ezdvpp"
        return 1
    fi
    return 0
}

check_param_configure()
{
    for i in `cat ${app_path}/param_configure.conf | awk -F'[ =]+' '{print $2}'`
    do
        if [[ ${i} = "" ]];then
            echo "please check your param_configure.conf to make sure that each parameter has a value"
            return 1
        fi
    done 

    #get and check format of remost_host ip
    check_remote_host
    if [ $? -ne 0 ];then
		return 1
    fi

    model_name=`cat ${app_path}/param_configure.conf | grep "model_name" | awk -F'[ =]+' '{print $2}'`
    [[ ${model_name##*.} == "om" ]] || (echo "please check your param_configure.conf to make sure that model_name has a valid name.";return 1)
    return 0
}

function main()
{
	check_param_configure
    if [ $? -ne 0 ];then
        return 1
    fi
    
	if tmp=`wc -l ${script_path}/Tag 2>/dev/null`;then
        line=`echo $tmp | awk -F' ' '{print $1}'`
        if [[ $line -ne 1 ]];then
            rm -rf ${script_path}/Tag
            build_common
            if [ $? -ne 0 ];then
                echo "ERROR: Failed to deploy common lib"
                return 1
            else
                echo "success" > ${script_path}/Tag
            fi
        else
            [[ "success" = `cat ${script_path}/Tag | grep "^success$"` ]] || build_common
            if [ $? -ne 0 ];then
                echo "ERROR: Failed to deploy common lib"
                return 1
            else
                echo "success" > ${script_path}/Tag
            fi
        fi
    else
        build_common
        if [ $? -ne 0 ];then
            echo "ERROR: Failed to deploy common lib"
            return 1
        else
            echo "success" > ${script_path}/Tag
        fi
    fi
    
    echo "Modify param information in graph.config..."
    count=0
    for om_name in $(find ${script_path}/ -name "${model_name}");do
		let count++
		if [ $count -ge 1 ];then
			break
		fi
    done
    
    if [ $count -eq 0 ];then
        echo "please push your model file in sample_classification/script/ "
        return 1
    fi
    om_name=$(basename ${om_name})
    cp ${script_path}/graph.template ${app_path}/graph.config
    sed -i "s#\${MODEL_PATH}#../../script/${om_name}#g"  ${app_path}/graph.config
    if [ $? != 0 ];then
		echo "gengrate graph.config error !"
		return 1
    fi 
    return 0	
}
main

