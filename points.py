from PIL import Image, ImageDraw
import random
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def generatePoints():
    radius = 5
    rangeX = (0, 512)
    rangeY = (0, 512)
    qty = 150  # or however many points you want

    # Generate a set of all points within 200 of the origin, to be used as offsets later
    # There's probably a more efficient way to do this.
    deltas = set()
    for x in range(-radius, radius+1):
        for y in range(-radius, radius+1):
            if x*x + y*y <= radius*radius:
                deltas.add((x,y))

    randPoints = []
    excluded = set()
    i = 0
    while i<qty:
        x = random.randrange(*rangeX)
        y = random.randrange(*rangeY)
        if (x,y) in excluded: continue
        randPoints.append((x,y))
        i += 1
        excluded.update((x+dx, y+dy) for (dx,dy) in deltas)
    return randPoints


def check(img):
    checklist=[]
    width,height=img.size
    # for x in range(width):
        # for y in range(height):
            # checklist.append(img.getcoord((x,y)))
    
    # print(checklist)
         
def mean(coords):
    # data = [[1, 2, 2], [1, 2, 1], [1, 1, 1]]
    mean = [sum(x)/len(x) for x in zip(*coords)]
    return mean

def cartesian_to_polar(x, y):
    r = math.sqrt(x ** 2 + y ** 2)
    theta = math.atan2(y, x)
    # theta=math.degrees(theta)
    return r, theta

def polar_coordinates(mean, rpts):
    # Load the image
    # image = Image.open("gen_img.png")

    # Determine the coordinates of the specific point (center)
    # center_pixel_value = image[center_y, center_x]
    mean_x,mean_y=mean

    new_cartesians=[]
    new_polars = []
    for pixel_coord in rpts:
        x, y = pixel_coord[0] - mean_x, pixel_coord[1] - mean_y
        r, theta = cartesian_to_polar(x, y)
        new_cartesians.append((x,y))
        new_polars.append((r,theta))

    return new_cartesians,new_polars

def drawLine(img_path,mean, new_coords):
    img=Image.open(img_path)
    draw=ImageDraw.Draw(img)
    m_x,m_y=mean
    for coord in new_coords:
        x1,y1=coord
        x,y=x1+m_x,y1+m_y
        coordinate=(x,y)
        # print(m_x,m_y,x,y)
        draw.line([m_x,m_y,x,y],fill='red',width=0)
    img.save("./points.png")

def draw_point_with_width(draw, rpts, width, color):
    # draw = ImageDraw.Draw(image)

    for x,y in rpts:
        # Calculate the coordinates of the rectangle or circle
        x1, y1 = x - width // 2, y - width // 2
        x2, y2 = x1 + width - 1, y1 + width - 1

        # Draw the rectangle or circle (you can choose one of the options)
        # draw.rectangle([x1, y1, x2, y2], fill=color)  # Option 1: Rectangle
        draw.ellipse([x1, y1, x2, y2], fill=color)   # Option 2: Circle

    # del draw



def cart2pol(pts):
    polar=[]
    for pt in pts:
        x,y=pt
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        polar.append((rho,phi))

    return(polar)

def best_fit_ellipse(pts):
    points=np.array(pts)
    points_float32=points.astype(np.float32)
    ellipse=cv2.fitEllipse(points_float32)
    print(points_float32)

    (center_x, center_y), (major_axis, minor_axis), angle = ellipse=ellipse

    # Create a scatter plot for the original points
    plt.scatter(points[:, 0], points[:, 1], label='Points')

    # Draw the best fit ellipse
    ellipse_patch = plt.Circle((center_x, center_y), major_axis / 2, angle=angle, fill=False, edgecolor='r', label='Best Fit Ellipse')
    plt.gca().add_patch(ellipse_patch)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Best Fit Ellipse')
    plt.legend()
    # plt.axis('equal')  # Set equal aspect ratio for both axes
    plt.grid(True)
    plt.savefig('best_fit_ellipse2.png')
    plt.show()




gen_img=Image.new('RGB',(512,512),(255,255,255))

draw=ImageDraw.Draw(gen_img)
rpts=generatePoints()
# draw.point(xy=rpts,fill=(0,0,0))
draw_point_with_width(draw,rpts,2,(0,0,0))

mean=mean(rpts)
draw.point(xy=mean,fill="red")

new_cartesians,new_polars=polar_coordinates(mean, rpts)

gen_img.save('gen_pts.png')
# print(new_cartesians)
drawLine('gen_pts.png',mean,new_cartesians)
print(new_cartesians)
print(new_polars)


# check(gen_img)
# bin_img=gen_img.convert('1')
# bin_img.save('gen_img.png')
# print("Break")
# check(bin_img)

# rpts=generatePoints()
# print(rpts) # these are self-generated points, write code to find points on image

# mean=mean(rpts)

# mean_coord=mean(rpts)
# print(mean_coord)

# print(cart2pol(rpts))
# print(bin_img.mode)


x=[item[0] for item in new_polars]
y=[item[1] for item in new_polars]

plt.scatter(x,y)
plt.xlabel("r")
plt.ylabel("theta")
plt.title("r vs theta")
plt.grid(True)
plt.savefig("r_vs_theta.png")

best_fit_ellipse(new_polars)