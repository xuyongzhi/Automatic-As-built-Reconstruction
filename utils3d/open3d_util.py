# 3 June
import numpy as np
import os,sys
import open3d
import matplotlib.pyplot as plt
import matplotlib.animation as anm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def gen_animation(pcds, ani_fn, ani_size):
    if not os.path.exists("./image/"):
        os.makedirs("./image/")

    k = 20
    s = 5.0
    gen_animation.index = -1
    gen_animation.imgs = []

    def rotate_view(vis):
        render_op = vis.get_render_option()
        render_op.point_size = 0
        render_op.mesh_show_back_face = True
        gen_animation.index = gen_animation.index + 1
        i = gen_animation.index
        print(f'i: {gen_animation.index}')
        ctr = vis.get_view_control()
        #ctr.change_field_of_view(-20)
        #view = ctr.get_field_of_view()
        #print(f'v:{view}')
        if i == 0:
          ctr.rotate(-s*k, 0.0)
        elif i >= 1 and i <=2*k:
          ctr.rotate(s, 0.0)
        elif i > 2*k and i <=3*k:
          ctr.rotate(-s, 0.0)
        elif i > 3*k and i <=4*k:
          ctr.rotate(0.0, s)
        elif i > 4*k and i <=6*k:
          ctr.rotate(0.0, -s)
        else:
          vis.register_animation_callback(None)

        #ctr.scale(0.1)

        image = vis.capture_screen_float_buffer(False)
        image = np.asarray(image)
        gen_animation.imgs.append(image)

        return False

    open3d.visualization.draw_geometries_with_animation_callback(pcds, rotate_view)

    fig = plt.figure()

    ims = []
    for i, img in enumerate( gen_animation.imgs[1:] ):

      #im = plt.imshow(img, animated=True)
      #plt.show()

      if ani_size is not None:
        h0,h1, w0,w1 = ani_size
        img = img[h0:h1,w0:w1,:]
      if i<2:
        img_fn = "./image/{:05d}.png".format(i)
        plt.imsave(img_fn, img, dpi = 1)
      im = plt.imshow(img, animated=True)
      #plt.show()
      ims.append([im])

    Writer = anm.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='xyz'), bitrate=None)
    ani = anm.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    h,w = img.shape[0:2]
    fig.set_size_inches(50, 50.0*h/w, True)
    plt.axis('off')
    #plt.show()

    ani.save(ani_fn, writer=writer)
    print(f'animation saved: {ani_fn}')


def draw_cus(models):
    open3d.visualization.RenderOption.line_width=2
    open3d.visualization.RenderOption.mesh_show_back_face=True
    open3d.visualization.RenderOption.show_coordinate_frame=True
    open3d.visualization.draw_geometries(models)

    return

    #open3d.visualization.RenderOption.line_width=5
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    for m in models:
        vis.add_geometry(m)
    #vis.get_render_option().line_width = 3
    vis.get_render_option().load_from_json(f"{BASE_DIR}/renderoption.json")
    vis.run()
    #print(f'point size: {vis.get_render_option().point_size}')
    #print(f'line width: {vis.get_render_option().line_width}')
    vis.destroy_window()
    pass

