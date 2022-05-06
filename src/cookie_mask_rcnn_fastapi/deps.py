import cookie_mask_rcnn as cmr
import cookie_mask_rcnn_fastapi as cmr_fapi


PRED_MODEL = cmr.modeling.utils.load_model(
    cmr_fapi.config.SETTINGS.PRED_MODEL_PATH)
