
class SUNCG_METAS:
    class_2_label0 = {'background':0,'wall':1, 'window':2, 'door':3,
                    'ceiling':5, 'floor': 4, 'room':6}
    label_2_class0 = {c:o for c,o in zip(class_2_label0.values(), class_2_label0.keys())}
    num_classes0 = len(class_2_label0)
    classes_order = []
    for l in range(num_classes0):
        classes_order.append( label_2_class0[l] )

    def __init__(self, classes):
        assert 'background' in classes
        if classes is None:
            classes = self.classes_order
        self.classes = classes
        class_2_label = {}
        label_2_class = {}

        for c in classes:
            assert c in self.classes_order, f"{c} is not a valid class name"

        l = 0
        for c in self.classes_order:
            if c in classes:
                class_2_label[c] = l
                label_2_class[l] = c
                l += 1
        #print(f'class_2_label: {class_2_label}')
        self.num_classes = len(classes)
        self.class_2_label = class_2_label
        self.label_2_class = label_2_class

    def label_num(self):
      return len(self.label_2_class)

    def show_colors(self):
      from utils3d.color_list import show_class_colors
      classes = self.classes
      classes[0] = 'missed gt,\nincorrect pred'
      show_class_colors(classes)

SUNCG_METAS0 = SUNCG_METAS(SUNCG_METAS.classes_order)

if __name__ == '__main__':
    dset_metas = SUNCG_METAS(['background', 'wall', 'window', 'door'])
    dset_metas.show_colors()
