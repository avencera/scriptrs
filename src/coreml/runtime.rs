use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;

use block2::RcBlock;
use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray,
};
use objc2_foundation::{NSCopying, NSMutableDictionary, NSString, NSURL};

use crate::coreml::array::{
    extract_output, multi_array_f32, multi_array_f32_cached, multi_array_i32,
    multi_array_i32_cached,
};
use crate::coreml::{CachedCoreMlInput, CoreMlInput, CoreMlTensor};
use crate::error::TranscriptionError;

pub(super) fn load_model(
    path: &Path,
    compute_units: MLComputeUnits,
) -> Result<Retained<MLModel>, TranscriptionError> {
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
    input_dict: &NSMutableDictionary<NSString, AnyObject>,
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
    inputs: &[CoreMlInput<'_>],
    output_names: &[&str],
) -> Result<HashMap<String, CoreMlTensor>, TranscriptionError> {
    autoreleasepool(|_| {
        let mut arrays = Vec::with_capacity(inputs.len());

        for input in inputs {
            let (name, array) = match input {
                CoreMlInput::F32 {
                    name,
                    values,
                    shape,
                } => (*name, multi_array_f32(values, shape, deallocator)?),
                CoreMlInput::I32 {
                    name,
                    values,
                    shape,
                } => (*name, multi_array_i32(values, shape, deallocator)?),
            };
            let key = NSString::from_str(name);
            let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*key);
            insert_input_feature(input_dict, key_copy, &array);
            arrays.push(array);
        }

        let output_provider = predict_features(
            model,
            ProtocolObject::from_ref(&*build_feature_provider(input_dict)?),
        )?;
        let mut outputs = HashMap::with_capacity(output_names.len());
        for output_name in output_names {
            let key = NSString::from_str(output_name);
            let array = output_multi_array(&output_provider, &key, output_name)?;
            let (data, shape) = extract_output(&array)?;
            outputs.insert((*output_name).to_owned(), CoreMlTensor { data, shape });
        }

        Ok(outputs)
    })
}

pub(super) fn insert_cached_input(
    input_dict: &NSMutableDictionary<NSString, AnyObject>,
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
    input: CachedCoreMlInput<'_>,
) -> Result<Retained<MLMultiArray>, TranscriptionError> {
    let (cached, array) = match input {
        CachedCoreMlInput::F32 { cached, values } => {
            debug_assert_eq!(values.len(), cached.total_elements);
            (cached, multi_array_f32_cached(values, cached, deallocator)?)
        }
        CachedCoreMlInput::I32 { cached, values } => {
            debug_assert_eq!(values.len(), cached.total_elements);
            (cached, multi_array_i32_cached(values, cached, deallocator)?)
        }
    };
    let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*cached.name);
    insert_input_feature(input_dict, key_copy, &array);
    Ok(array)
}

pub(super) fn insert_input_feature(
    input_dict: &NSMutableDictionary<NSString, AnyObject>,
    key_copy: &ProtocolObject<dyn NSCopying>,
    multi_array: &MLMultiArray,
) {
    // SAFETY: multi_array is a live CoreML object for this prediction call, and setObject retains
    // SAFETY: the inserted feature value before the temporary Retained<MLFeatureValue> is dropped
    unsafe {
        let feature_value = MLFeatureValue::featureValueWithMultiArray(multi_array);
        input_dict.setObject_forKey(feature_value_as_any_object(&feature_value), key_copy);
    }
}

pub(super) fn build_feature_provider(
    input_dict: &NSMutableDictionary<NSString, AnyObject>,
) -> Result<Retained<MLDictionaryFeatureProvider>, TranscriptionError> {
    // SAFETY: input_dict only contains NSString keys and MLFeatureValue-backed Objective-C objects
    unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            input_dict,
        )
    }
    .map_err(|error| TranscriptionError::CoreMl(format!("feature provider failed: {error}")))
}

pub(super) fn predict_features(
    model: &MLModel,
    input_ref: &ProtocolObject<dyn MLFeatureProvider>,
) -> Result<Retained<ProtocolObject<dyn MLFeatureProvider>>, TranscriptionError> {
    // SAFETY: input_ref is a valid feature provider built from live Objective-C objects
    unsafe { model.predictionFromFeatures_error(input_ref) }
        .map_err(|error| TranscriptionError::CoreMl(format!("prediction failed: {error}")))
}

pub(super) fn output_multi_array(
    output: &ProtocolObject<dyn MLFeatureProvider>,
    output_key: &NSString,
    output_name: &str,
) -> Result<Retained<MLMultiArray>, TranscriptionError> {
    // SAFETY: output is a retained CoreML feature provider produced by a successful prediction call
    let feature = unsafe { output.featureValueForName(output_key) }.ok_or_else(|| {
        TranscriptionError::CoreMl(format!("missing CoreML output `{output_name}`"))
    })?;
    // SAFETY: output_key names the declared tensor output for this model
    unsafe { feature.multiArrayValue() }.ok_or_else(|| {
        TranscriptionError::CoreMl(format!("CoreML output `{output_name}` was not an array"))
    })
}

fn feature_value_as_any_object(feature_value: &MLFeatureValue) -> &AnyObject {
    // SAFETY: MLFeatureValue is an Objective-C object and shares pointer representation
    // SAFETY: with AnyObject for APIs that erase the concrete class type
    unsafe { &*(feature_value as *const MLFeatureValue).cast::<AnyObject>() }
}
