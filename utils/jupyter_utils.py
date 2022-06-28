def one_grid_computation(prediction, target, mask, grid, criterion, criterion_base, target_len):
    gridI = grid.ith
    SX, EX, SY, EY = grid.get_grid_bbox()
    prediction = prediction[..., SX:EX, SY:EY]
    target = target[..., SX:EX, SY:EY]
    mask = mask[..., SX:EX, SY:EY]
    loss_dict = {gridI: {}}
    loss_base_dict = {gridI: {}}
    pred_dict = {gridI: {}}
    tar_dict = {gridI: {}}
    for t in range(target_len):
        # prediction is different from target in terms of dimensions. 1st dim is time
        prediction_t = prediction[t:t + 1, ...].contiguous()

        target_t = target[:, t:t + 1, :, :].contiguous()
        mask_t = mask[:, t:t + 1, :, :].contiguous()
        loss_dict[gridI][t] = criterion(prediction_t, target_t, mask_t).item()
        loss_base_dict[gridI][t] = criterion_base(prediction_t, target_t, mask_t).item()

        prediction_t = prediction_t.permute(1, 0, 2, 3)
        prediction_t = prediction_t.cpu().numpy().reshape(-1, )
        target_t = target_t.cpu().numpy().reshape(-1, )
        pred_dict[gridI][t] = prediction_t
        tar_dict[gridI][t] = target_t

    return {'loss': loss_dict, 'base_loss': loss_base_dict, 'prediction': pred_dict, 'target': tar_dict}
