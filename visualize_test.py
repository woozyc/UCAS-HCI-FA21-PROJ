import matplotlib.pyplot as plt
import numpy as np
import h5py

#define connectivities between joints
#for each entry, define as [joint_start, joint_end, left(1) or right(0)]
skeleton_connectivity = [[0, 1, 0], [1, 2, 0], [2, 6, 0], [5, 4, 1],
                         [4, 3, 1], [3, 6, 1], [6, 7, 0], [7, 8, 0],
                         [8, 16, 0], [9, 16, 0], [8, 12, 0], [11, 12, 0],
                         [10, 11, 0], [8, 13, 1], [13, 14, 1], [14, 15, 1]]

def draw3Dpose(pos3D, ax, lcolor="#3498db", rcolor="#e74c3c"):
    for i in skeleton_connectivity:
        #x, y and z are coordinates of a 'bone' line segment
        x = np.array([pos3D[i[0], 0], pos3D[i[1], 0]])
        y = np.array([pos3D[i[0], 1], pos3D[i[1], 1]])
        z = np.array([pos3D[i[0], 2], pos3D[i[1], 2]])
        ax.plot(x, y, z, lw=2, c=lcolor if i[2] else rcolor)
    
    #size of the scene, choose left ankle as bottom center
    room_size = 800
    xroot, yroot, zroot = pos3D[5, 0], pos3D[5, 1], pos3D[5, 2]
    ax.set_xlim3d([-room_size + xroot, room_size + xroot])
    ax.set_zlim3d([0, 2 * room_size + zroot])
    ax.set_ylim3d([-room_size + yroot, room_size + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
if __name__ == '__main__':
                                     #RAnkle RKnee RHip LHip LKnee LAnkle
    specific_3d_skeleton = np.array([[-60, 400, 110],
                                     [-20, 350, 540],
                                     [20, 380, 990],
                                     [310, 330, 970],
                                     [280, 340, 515],
                                     [255, 400, 80],
                                     #Pelvis Spine Thorax HeadTop
                                     [160, 350, 980],
                                     [180, 335, 1240],
                                     [190, 290, 1490],
                                     [150, 210, 1680],
                                     #RWrist RElbow RShoulder LShoulder LElbow LWrist Nose
                                     [-380, 410, 1210],
                                     [-170, 485, 1300],
                                     [35, 360, 1460],
                                     [350, 285, 1455],
                                     [610, 260, 1320],
                                     [778, 90, 1260],
                                     [165, 206, 1570]])
    start_3d_skeleton = np.array(   [[ -96.19869995,  154.22174072,   74.16516876],
                                     [ -82.29842377,   52.99495316,  531.51477051],
                                     [ -70.89446259,   33.29279709, 1003.15881348],
                                     [ 173.97718811,   44.24522018,  964.47912598],
                                     [ 224.32675171,   36.86126328,  495.03509521],
                                     [ 278.90710449,  124.74990082,   37.97391891],
                                     [  51.54119873,   38.76900101,  983.81896973],
                                     [  58.8880806 ,   57.2521286 , 1239.33227539],
                                     [  53.24123001,   42.16963959, 1489.1706543 ],
                                     [  48.84748459,  -34.13542175, 1691.88195801],
                                     [-239.15065002,   37.90293884,  913.34631348],
                                     [-207.40151978,  115.27999115, 1147.89526367],
                                     [-117.49203491,   57.29710388, 1424.22851562],
                                     [ 224.95596313,   72.51789856, 1432.69909668],
                                     [ 302.10842896,  111.42869568, 1149.25720215],
                                     [ 306.06100464,   23.35810089,  916.36981201],
                                     [  55.14346695,  -40.15619278, 1577.21533203]])
    start_3d_skeleton_2 = np.array( [[-227.96409607, -153.33421326,   78.07400513],
                                     [-213.52304077, -261.86239624,  533.72937012],
                                     [-199.20167542, -289.77844238, 1004.87890625],
                                     [  46.51939774, -296.18557739,  970.84503174],
                                     [  59.91590118, -276.02877808,  499.27157593],
                                     [  74.1147995 , -169.56407166,   43.12247467],
                                     [ -76.34130096, -292.98199463,  987.86199951],
                                     [ -67.32461548, -265.1809082 , 1242.47473145],
                                     [ -64.27209473, -277.38711548, 1492.51538086],
                                     [ -50.73743439, -338.23135376, 1702.96289062],
                                     [-341.92211914, -287.26196289,  893.17059326],
                                     [-294.95602417, -213.88479614, 1126.44812012],
                                     [-227.06987   , -251.23179626, 1412.45947266],
                                     [  98.65423584, -258.19094849, 1410.76867676],
                                     [ 174.15899658, -255.87852478, 1124.23852539],
                                     [ 185.49449158, -354.52090454,  895.87524414],
                                     [ -60.88600159, -349.60873413, 1588.98095703]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=22, azim=-130)
    #draw3Dpose(specific_3d_skeleton, ax)
    #plt.show()
    #f = h5py.File('D:\\Downloads\\h36m\\h36m\\S9\\MyPoses\\3D_positions\\Waiting.h5', 'r')
    f = h5py.File('D:\\Downloads\\h36m\\h36m\\S9\\MyPoses\\3D_positions\\Phoning.h5', 'r')
    print(f.keys())
    for key in f.keys():
        print(f[key].name)
        print(f[key].shape)
        #print(f[key].value[0])
    #RAnkle RKnee RHip LHip LKnee LAnkle Pelvis Spine Thorax Head RWrist RElbow RShoulder LShoulder LElbow LWrist Nose
    valid_joints = [3,2,1,6,7,8,0,12,13,15,27,26,25,17,18,19,14]
    poses_world = np.array(f['3D_positions']).T.reshape(-1, 32, 3)
    print(np.shape(poses_world))

    plt.ion()
    for i in range(1):
        for j in range(len(poses_world)):
        #for j in range(0,2):
            #print(j)
            ax.lines = []
            pose = poses_world[j][valid_joints]
            print(pose)
            draw3Dpose(start_3d_skeleton, ax)
            plt.pause(0.001)
    plt.ioff()
    plt.show()