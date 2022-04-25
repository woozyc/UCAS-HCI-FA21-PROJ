import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

#define connectivities between joints
#for each entry, define as [joint_start, joint_end, color]
skeleton_connectivity = [[0, 1, 0], [1, 2, 0], [2, 6, 0], [5, 4, 0],
                         [4, 3, 0], [3, 6, 0], [6, 7, 0], [7, 8, 0],
                         [8, 16, 0], [9, 16, 0], [8, 12, 0], [11, 12, 1],
                         [10, 11, 1], [8, 13, 0], [13, 14, 0], [14, 15, 0]]

def draw3Dpose(pos3D, ax, lcolor="#e74c3c", rcolor="#3498db"):
    for i in skeleton_connectivity:
        #x, y and z are coordinates of a 'bone' line segment
        x = np.array([pos3D[i[0], 0], pos3D[i[1], 0]])
        y = np.array([pos3D[i[0], 1], pos3D[i[1], 1]])
        z = np.array([pos3D[i[0], 2], pos3D[i[1], 2]])
        ax.plot(x, y, z, lw=2, c=lcolor if i[2] else rcolor)
    
    #size of the scene, choose left ankle as bottom center
    room_size = 750
    xroot, yroot, zroot = pos3D[5, 0], pos3D[5, 1], pos3D[5, 2]
    ax.set_xlim3d([-room_size + xroot, room_size + xroot])
    ax.set_zlim3d([0, 2 * room_size + zroot])
    ax.set_ylim3d([-room_size + yroot, room_size + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
if __name__ == '__main__':
    #use 2 default upstanding human poses as start pos
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
    
    #read csv file, setup data structure
    array = np.genfromtxt("D:\\Code\\Python\\hum-com-inter\\testdata.csv", delimiter=',')
    w = array[1:, 13]
    x = array[1:, 14]
    y = array[1:, 15]
    z = array[1:, 16]
    Rs = []
    print(np.shape(array))
    print(np.shape(w))
    for i in range(len(w)):
        q = np.array([w[i], x[i], y[i], z[i]])
        n = np.dot(q, q)
        #print(np.sqrt(2.0 / n))
        q = q * np.sqrt(2.0 / n)
        q = np.outer(q, q)
        Rs.append(np.array([[1.0 - q[2,2] - q[3,3], q[1,2] + q[3,0], q[1,3] - q[2,0]],
                            [q[1,2] - q[3,0], 1.0 - q[1,1] - q[3,3], q[2,3] + q[1,0]],
                            [q[1,3] + q[2,0], q[2,3] - q[1,0], 1.0 - q[1,1] - q[2,2]]]))

    def quaternion_multiply(Q0,Q1):
        """
        Multiplies two quaternions.
    
        Input
        :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31) 
        :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32) 
    
        Output
        :return: A 4 element array containing the final quaternion (q03,q13,q23,q33) 
    
        """
        # Extract the values from Q0
        w0 = Q0[0]
        x0 = Q0[1]
        y0 = Q0[2]
        z0 = Q0[3]

        # Extract the values from Q1
        w1 = Q1[0]
        x1 = Q1[1]
        y1 = Q1[2]
        z1 = Q1[3]

        # Computer the product of the two quaternions, term by term
        Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

        # Create a 4 element array containing the final quaternion
        final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])

        # Return a 4 element array containing the final quaternion (q02,q12,q22,q32) 
        return final_quaternion
    
    #aign to world coordinate, find rotate axis
    q_0 = np.array([w[0], x[0], y[0], z[0]])
    q_0_inv = np.array([w[0], -x[0], -y[0], -z[0]])
    q_0_inv = q_0_inv / np.linalg.norm(q_0_inv)
    max_len = 0
    for i in range(len(w)):
        q_temp = np.array([w[i], x[i], y[i], z[i]])
        temp_len = np.linalg.norm(q_temp - q_0)
        if temp_len > max_len:
            max_len = temp_len
            max = i
    print(max)
    q_final = np.array([w[max], x[max], y[max], z[max]])
    q_final = q_final / np.linalg.norm(q_final)
    q_relative = quaternion_multiply(q_0_inv, q_final)
    q_theta = 2 * math.acos(q_relative[0])
    #z coordinate is already aligned
    rotate_axis = np.array([q_relative[1], q_relative[2], q_relative[3]]) / math.sin(q_theta/2)
    print(q_relative)
    print(rotate_axis)
    shoulder_line = start_3d_skeleton[13] - start_3d_skeleton[12]
    #normalize
    shoulder_line = shoulder_line / np.linalg.norm(shoulder_line)
    rotate_axis = rotate_axis / np.linalg.norm(rotate_axis)
    print(shoulder_line)
    print(rotate_axis)
    #relative rotation
    theta = math.acos(np.dot(shoulder_line, rotate_axis))
    rotate_relavite = np.array([shoulder_line[1]*rotate_axis[2]-shoulder_line[2]*rotate_axis[1],
                                shoulder_line[2]*rotate_axis[0]-shoulder_line[0]*rotate_axis[2],
                                shoulder_line[0]*rotate_axis[1]-shoulder_line[1]*rotate_axis[0]])
    #shoulder_angle = math.acos(shoulder_line[0]) if (shoulder_line[1]) > 0 else -math.acos(shoulder_line[0])
    #rotate_angle = -math.acos(rotate_axis[0]) if (rotate_axis[1]) > 0 else math.acos(rotate_axis[0])
    #R_angle = shoulder_angle - rotate_angle
    #R_world2sensor = np.array([[math.cos(R_angle), -math.sin(R_angle), 0],
    #                    [math.sin(R_angle),  math.cos(R_angle), 0],
    #                    [0, 0, 1]])
    #print("shoulder:", shoulder_angle* 180.0/math.pi, "rotate:", rotate_angle* 180.0/math.pi, "R:", R_angle* 180.0/math.pi)
    costheta = math.cos(q_theta)
    sintheta = math.sin(q_theta)
    ux = rotate_relavite[0]
    uy = rotate_relavite[1]
    uz = rotate_relavite[2]
    R_world2sensor = np.array(
                            [[costheta + (1-costheta)*ux*ux, ux*uy*(1-costheta)-uz*sintheta, ux*uz*(1-costheta)+uy*sintheta],
                            [uy*ux*(1-costheta)+uz*sintheta, costheta+uy*uy*(1-costheta), uy*uz*(1-costheta)-ux*sintheta],
                            [uz*ux*(1-costheta)-uy*sintheta, uz*uy*(1-costheta)+ux*sintheta, costheta+uz*uz*(1-costheta)]])

    #draw
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(elev=15, azim=-140)
    axs = []
    plt.ion()
    maxangle = 0.0
    for i in range(1):
        #for j in range(len(w)):
        for j in range(50, 340, 2):
            if j % 10 == 0:
                print(j)
            ax.lines = []
            skeleton = start_3d_skeleton.copy()    #start world_position
            #use quaternion to compute rotate axis and angle
            q_temp = np.array([w[j], x[j], y[j], z[j]])
            q_temp = q_temp / np.linalg.norm(q_temp)
            if j == 170:
                print("q_temp:", q_temp)
            q_relative = quaternion_multiply(q_0_inv, q_temp)
            q_relative = q_relative / np.linalg.norm(q_relative)
            #print("q_0_inv:", q_0_inv)
            if j == 170:
                print("q_relative:", q_relative)
            q_theta = 2 * math.acos(q_relative[0])
            if q_theta > maxangle:
                maxangle = q_theta
            div = math.sin(q_theta/2)
            rotate_axis_sensor = np.array([[q_relative[1]], [q_relative[2]], [q_relative[3]]]) / (div if div else 0.001)
            rotate_axis_sensor = rotate_axis_sensor / np.linalg.norm(rotate_axis_sensor)
            if j == 170:
                print("rotate_axis_sensor:", rotate_axis_sensor, "angle:",q_theta * 180.0/math.pi)
            #transform to world coordinate
            rotate_axis_world = R_world2sensor.dot(rotate_axis_sensor).reshape(3, 1)
            #ax.plot(np.array([0, rotate_axis_world[0][0]*500]), np.array([0, rotate_axis_world[1][0]*500]), np.array([0, rotate_axis_world[2][0]*500]), lw=2)
            #print("rotate_axis_world:", rotate_axis_world)
            #rotate_axis_world = rotate_axis_world + skeleton[12]
            #print("skeleton:", skeleton[12])
            #print("skeleton:", skeleton[12].reshape(3,1))
            #print("rotate_axis_world:", rotate_axis_world)
            #calculate skeleton new coordinate
            rotate_axis_world = rotate_axis_world.reshape(3,)
            norm = np.linalg.norm(rotate_axis_world)
            rotate_axis_world = rotate_axis_world / norm
            costheta = math.cos(q_theta)
            sintheta = math.sin(q_theta)
            ux = rotate_axis_world[0]
            uy = rotate_axis_world[1]
            uz = rotate_axis_world[2]
            #print("rotate_axis_world:", rotate_axis_world)
            Rotate_world = np.array(
                            [[costheta + (1-costheta)*ux*ux, ux*uy*(1-costheta)-uz*sintheta, ux*uz*(1-costheta)+uy*sintheta],
                            [uy*ux*(1-costheta)+uz*sintheta, costheta+uy*uy*(1-costheta), uy*uz*(1-costheta)-ux*sintheta],
                            [uz*ux*(1-costheta)-uy*sintheta, uz*uy*(1-costheta)+ux*sintheta, costheta+uz*uz*(1-costheta)]])
            #print("rotate matrix:", Rotate_world)
            up_arm_dir = skeleton[11] - skeleton[12]
            low_arm_dir = skeleton[10] - skeleton[12]
            #print("up_arm_dir:", up_arm_dir)
            up_arm_dir = Rotate_world.dot(up_arm_dir.reshape(3,1)).reshape(3,)
            low_arm_dir = Rotate_world.dot(low_arm_dir.reshape(3,1)).reshape(3,)
            #print("up_arm_dir:", up_arm_dir)
            skeleton[10] = skeleton[12] + low_arm_dir
            skeleton[11] = skeleton[12] + up_arm_dir
            #print("elbow:", skeleton[11], "hand:",skeleton[10])
            draw3Dpose(skeleton, ax)
            ax.plot(np.array([start_3d_skeleton[12][0], start_3d_skeleton[11][0]]), np.array([start_3d_skeleton[12][1], start_3d_skeleton[11][1]]), np.array([start_3d_skeleton[12][2], start_3d_skeleton[11][2]]), lw=1.3, c='y')
            ax.plot(np.array([start_3d_skeleton[11][0], start_3d_skeleton[10][0]]), np.array([start_3d_skeleton[11][1], start_3d_skeleton[10][1]]), np.array([start_3d_skeleton[11][2], start_3d_skeleton[10][2]]), lw=1.3, c='y')
            #axs.append(ax)
            plt.pause(0.001)
    print(maxangle * 180.0/math.pi)
    plt.ioff()
    plt.show()
    #ani = animation.ArtistAnimation(fig, axs, interval=50) #生成动画
    #ani.save("test.gif", writer='pillow')