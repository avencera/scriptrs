use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;

use block2::RcBlock;
use objc2::AnyThread;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration,
};
use objc2_foundation::{NSCopying, NSMutableDictionary, NSString, NSURL};

use crate::coreml::array::{extract_output, multi_array_f32, multi_array_i32};
use crate::coreml::{CoreMlInput, CoreMlTensor};
use crate::error::TranscriptionError;

pub(super) fn load_model(
    path: &Path,
    compute_units: MLComputeUnits,
) -> Result<objc2::rc::Retained<MLModel>, TranscriptionError> {
    let path_str = NSString::from_str(&path.to_string_lossy());
    let url = NSURL::fileURLWithPath_isDirectory(&path_str, true);
    // SAFETY: the URL and configuration are valid Objective-C objects for the duration
    // SAFETY: of this call, and CoreML retains anything it needs internally
    unsafe {
        let config = MLModelConfiguration::new();
        config.setComputeUnits(compute_units);
        MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
    }
    .map_err(|error| TranscriptionError::CoreMl(format!("failed to load model: {error}")))
}

pub(super) fn predict(
    model: &MLModel,
    inputs: &[CoreMlInput<'_>],
    output_names: &[&str],
) -> Result<HashMap<String, CoreMlTensor>, TranscriptionError> {
    let noop_deallocator = RcBlock::new(|_: NonNull<c_void>| {});
    let input_dict = NSMutableDictionary::<NSString, AnyObject>::new();
    let mut arrays = Vec::with_capacity(inputs.len());

    for input in inputs {
        let (name, array) = match input {
            CoreMlInput::F32 {
                name,
                values,
                shape,
            } => (*name, multi_array_f32(values, shape, &noop_deallocator)?),
            CoreMlInput::I32 {
                name,
                values,
                shape,
            } => (*name, multi_array_i32(values, shape, &noop_deallocator)?),
        };
        let key = NSString::from_str(name);
        let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*key);
        // SAFETY: the feature value wraps the live MLMultiArray for this call, and the
        // SAFETY: dictionary retains inserted Objective-C objects before returning
        unsafe {
            let feature_value = MLFeatureValue::featureValueWithMultiArray(&array);
            input_dict.setObject_forKey(feature_value_as_any_object(&feature_value), key_copy);
        }
        arrays.push(array);
    }

    // SAFETY: the dictionary contains NSString keys and MLFeatureValue-backed objects,
    // SAFETY: which is exactly what MLDictionaryFeatureProvider expects
    let input_provider = unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &input_dict,
        )
    }
    .map_err(|error| TranscriptionError::CoreMl(format!("feature provider failed: {error}")))?;

    let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
        ProtocolObject::from_ref(&*input_provider);
    // SAFETY: `input_ref` is a valid feature provider built from the live dictionary above
    let output_provider = unsafe { model.predictionFromFeatures_error(input_ref) }
        .map_err(|error| TranscriptionError::CoreMl(format!("prediction failed: {error}")))?;

    let mut outputs = HashMap::with_capacity(output_names.len());
    for output_name in output_names {
        let key = NSString::from_str(output_name);
        // SAFETY: `key` names a declared output in the CoreML model when prediction succeeds
        let feature = unsafe { output_provider.featureValueForName(&key) }.ok_or_else(|| {
            TranscriptionError::CoreMl(format!("missing CoreML output `{output_name}`"))
        })?;
        // SAFETY: the retrieved feature value is expected to contain an MLMultiArray tensor
        let array = unsafe { feature.multiArrayValue() }.ok_or_else(|| {
            TranscriptionError::CoreMl(format!("CoreML output `{output_name}` was not an array"))
        })?;
        let (data, shape) = extract_output(&array)?;
        outputs.insert((*output_name).to_owned(), CoreMlTensor { data, shape });
    }

    Ok(outputs)
}

fn feature_value_as_any_object(feature_value: &MLFeatureValue) -> &AnyObject {
    // SAFETY: MLFeatureValue is an Objective-C object and shares pointer representation
    // SAFETY: with AnyObject for APIs that erase the concrete class type
    unsafe { &*(feature_value as *const MLFeatureValue).cast::<AnyObject>() }
}
