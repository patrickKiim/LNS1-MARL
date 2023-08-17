import subprocess
import os
import tempfile


def writeMap(mapFile, Map):
    mapFile.write("height " +str(Map.shape[0]) + "\n")
    mapFile.write("width " + str(Map.shape[1]) + "\n")
    for i in range(Map.shape[0]):
        for j in range(Map.shape[1]):
            if Map[i][j]==0:
                mapFile.write(".")
            else:
                mapFile.write("@")
        mapFile.write("\n")

def writeAgentPos(posFile, start_list,goal_list):
    for i in range(len(start_list)):
        posFile.write(str(start_list[i][0])+" "+str(start_list[i][1])+" "+str(goal_list[i][0])+" "+str(goal_list[i][1])+"\n")

def runPP(inputWorldFile,inputPosFile,outputFile,num_agent):
    argument = "./LNS_function1/function1 -m " + inputWorldFile + " -a " + inputPosFile + " -o " + outputFile + " -n " + str(num_agent)
    with tempfile.TemporaryFile() as tempf:  # generate temp file
        proc = subprocess.Popen(argument, stdout=tempf, shell=True)  # using shell to execute argment, return completedprocess object to tempf
        proc.wait()  # wait for the processing ned

def readResults(resultFile):
    all_lines=resultFile.readlines()
    paths=[]
    for i,line in enumerate(all_lines):
        if i==0:
            can_not_use=bool(int(line[0]))
        elif i==1:
            makespan=int(line[:-1])
        elif i==2:
            global_num_collison=int(line[:-1])
        else:
            positions=line.split("-")
            path=[]
            for j in positions:
                if j=="\n":
                    continue
                k=j.split(" ")
                path.append((int(k[0]),int(k[1])))
            paths.append(path)
    return can_not_use,makespan, global_num_collison, paths

def run_pp(Map,start_list,goal_list,env_id):
    curr_path=os.getcwd()
    inputWorldFile = curr_path+"/record_files/world"+str(env_id)+".txt"
    inputPosFile = curr_path+"/record_files/pos"+str(env_id)+".txt"
    outputFile = curr_path+"/record_files/pp_output"+str(env_id)+".txt"
    mapFile = open(inputWorldFile, "w")
    posFile = open(inputPosFile, "w")
    writeMap(mapFile, Map)
    writeAgentPos(posFile, start_list,goal_list)
    mapFile.close()
    posFile.close()
    runPP(inputWorldFile,inputPosFile,outputFile,len(start_list))
    resultFile = open(outputFile, 'r')

    can_not_use, makespan, global_num_collison, paths = readResults(resultFile)

    resultFile.close()
    return can_not_use,makespan, global_num_collison, paths

def writePath(pathFile,destroy_weights, temp_path):
    pathFile.write(str(destroy_weights[0])+" "+str(destroy_weights[1])+" "+str(destroy_weights[2])+"\n")
    for i in temp_path:
        for t in i:
            pathFile.write(str(t[0])+" "+str(t[1])+"-")
        pathFile.write("\n")


def run_selection(inputWorldFile,inputPosFile,inputpathFile,outputFile,local_num_agents,ALNS,global_num_collison,update_weight,
                        selected_neighbor,num_agent):
    argument = "./LNS_function2/function2 -m " + inputWorldFile + " -a " + inputPosFile +" -p " + inputpathFile+" -o " + outputFile + " -n " + str(
        local_num_agents) +" -l "+str(ALNS)+" -c "+str(global_num_collison)+ " -u "+str(update_weight)+ " -s "+str(selected_neighbor)+" -g "+str(num_agent)
    with tempfile.TemporaryFile() as tempf:  # generate temp file
        proc = subprocess.Popen(argument, stdout=tempf,
                                shell=True)  # using shell to execute argment, return completedprocess object to tempf
        proc.wait()  # wait for the processing ned

def check_collision(path,num_agent,map_size,env_id):
    curr_path = os.getcwd()
    inputPathFile = curr_path + "/record_files/temp_paths" + str(env_id) + ".txt"
    outputFile = curr_path + "/record_files/check_output" + str(env_id) + ".txt"
    pathFile = open(inputPathFile, "w")

    for i in path:
        for t in i:
            pathFile.write(str(t[0])+" "+str(t[1])+"-")
        pathFile.write("\n")
    pathFile.close()

    argument = "./LNS_function3/function3 -m " + str(map_size) + " -p " + inputPathFile + " -o " + outputFile + " -n " + str(num_agent)
    with tempfile.TemporaryFile() as tempf:  # generate temp file
        proc = subprocess.Popen(argument, stdout=tempf, shell=True)
        proc.wait()

    resultFile = open(outputFile, 'r')
    all_lines=resultFile.readlines()
    true_dy_co=int(all_lines[0][:-1])
    resultFile.close()
    return true_dy_co

def readResults_2(resultFile):
    all_lines=resultFile.readlines()
    sipps_paths=[]
    for i,line in enumerate(all_lines):
        if i==0:
            global_num_collison=int(line[:-1])
        elif i==1:
            j=line.split(" ")
            destroy_weights=[float(j[0]),float(j[1]),float(j[2])]
        elif i==2:
            local_agents=[]
            j=line.split(" ")
            for k in j:
                if k=="\n":
                    continue
                local_agents.append(int(k))
        elif i==3:
            global_succ=bool(int(line[0]))
        elif i==4:
            selected_neighbor=int(line[0])
        elif i == 5:
            makespan=int(line[:-1])
        else:
            if global_succ==False:
                positions=line.split("-")
                path=[]
                for j in positions:
                    if j=="\n":
                        continue
                    k=j.split(" ")
                    path.append((int(k[0]),int(k[1])))
                sipps_paths.append(path)

    return global_num_collison, destroy_weights, local_agents, global_succ, selected_neighbor, makespan,sipps_paths

def adaptive_destroy(temp_path,local_num_agents,ALNS,global_num_collison,
                         destroy_weights,update_weight,selected_neighbor, env_id):
    curr_path = os.getcwd()
    inputWorldFile =curr_path+ "/record_files/world" + str(env_id) + ".txt"
    inputPosFile = curr_path+"/record_files/pos" + str(env_id) + ".txt"
    inputpathFile=curr_path+"/record_files/paths" + str(env_id) + ".txt"
    outputFile = curr_path+"/record_files/selection_output" + str(env_id) + ".txt"
    pathFile = open(inputpathFile, "w")
    writePath(pathFile,destroy_weights, temp_path)
    pathFile.close()
    run_selection(inputWorldFile,inputPosFile,inputpathFile,outputFile,local_num_agents,ALNS,global_num_collison,
                         update_weight,selected_neighbor,len(temp_path))
    resultFile = open(outputFile, 'r')

    global_num_collison, destroy_weights, local_agents, global_succ, selected_neighbor, makespan ,sipps_path = readResults_2(resultFile)

    resultFile.close()
    return global_num_collison, destroy_weights, local_agents, global_succ, selected_neighbor, makespan,sipps_path