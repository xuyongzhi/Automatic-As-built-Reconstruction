import torch, math

def yaw_direction_loss(input_yaw, anchor_yaw, size_average, weight=0.001):
    # target_yaw in [-pi/2,pi/2]
    # truth is 1
    pred_yaw = input_yaw + anchor_yaw
    direction_loss = torch.abs(pred_yaw) > math.pi * 0.5 + 1e-6
    direction_loss = direction_loss.float()
    print(f'direction_loss: {direction_loss}')
    print(f'pred_yaw: {pred_yaw}')
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    direction_loss *= weight

    if size_average:
      return direction_loss.mean()

    return direction_loss.sum()

