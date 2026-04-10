use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc;

use block2::RcBlock;
use eyre::{Result, bail, eyre};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLAllComputeDevices, MLCPUComputeDevice, MLComputeDeviceProtocol, MLComputePlan,
    MLComputePlanDeviceUsage, MLComputeUnits, MLGPUComputeDevice, MLModel, MLModelConfiguration,
    MLModelStructureProgram, MLModelStructureProgramFunction, MLNeuralEngineComputeDevice,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSString, NSURL};
use scriptrs::ModelBundle;

const ENCODER_DIR: &str = "parakeet-v2/encoder.mlmodelc";
const DECODER_DIR: &str = "parakeet-v2/decoder.mlmodelc";
const JOINT_DECISION_DIR: &str = "parakeet-v2/joint-decision.mlmodelc";
const VAD_DIR: &str = "vad/silero-vad.mlmodelc";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:?}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        bail!("CoreML compute inspection is only supported on macOS")
    }

    #[cfg(target_os = "macos")]
    {
        let args = Args::parse(env::args().skip(1))?;
        let bundle = resolve_bundle(&args)?;

        println!("models_root: {}", bundle.root().display());
        println!("compute_units: {}", compute_units_name(args.compute_units));
        print_device_list("coreml_available_devices", unsafe {
            MLModel::availableComputeDevices()
        });
        print_device_list("all_compute_devices", unsafe { MLAllComputeDevices() });

        inspect_model(
            "encoder",
            &bundle.root().join(ENCODER_DIR),
            args.compute_units,
            args.show_ops,
        )?;
        inspect_model(
            "decoder",
            &bundle.root().join(DECODER_DIR),
            args.compute_units,
            args.show_ops,
        )?;
        inspect_model(
            "joint_decision",
            &bundle.root().join(JOINT_DECISION_DIR),
            args.compute_units,
            args.show_ops,
        )?;

        if args.long_form {
            inspect_model(
                "vad",
                &bundle.root().join(VAD_DIR),
                args.compute_units,
                args.show_ops,
            )?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Args {
    models_dir: Option<PathBuf>,
    pretrained: bool,
    long_form: bool,
    compute_units: MLComputeUnits,
    show_ops: usize,
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut args = args.into_iter();
        let mut models_dir = None;
        let mut pretrained = false;
        let mut long_form = false;
        let mut compute_units = MLComputeUnits::CPUAndNeuralEngine;
        let mut show_ops = 0usize;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--models-dir" => models_dir = Some(next_path(&mut args, "--models-dir")?),
                "--pretrained" => pretrained = true,
                "--long-form" => long_form = true,
                "--compute-units" => {
                    let value = next_value(&mut args, "--compute-units")?;
                    compute_units = parse_compute_units(&value)?;
                }
                "--show-ops" => {
                    show_ops = next_value(&mut args, "--show-ops")?
                        .parse()
                        .map_err(|error| eyre!("invalid --show-ops value: {error}"))?;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                flag if flag.starts_with('-') => bail!("unknown flag: {flag}"),
                path => {
                    if models_dir.is_some() {
                        bail!("unexpected positional argument: {path}")
                    }
                    models_dir = Some(PathBuf::from(path));
                }
            }
        }

        if pretrained && models_dir.is_some() {
            bail!("use either --pretrained or --models-dir, not both")
        }

        Ok(Self {
            models_dir,
            pretrained,
            long_form,
            compute_units,
            show_ops,
        })
    }
}

fn print_usage() {
    eprintln!(
        "Usage:
  cargo run --example inspect_coreml_compute -- --models-dir <dir>
  cargo run --example inspect_coreml_compute -- --pretrained
  cargo run --example inspect_coreml_compute --features long-form -- --pretrained --long-form

Options:
  --models-dir <dir>         local scriptrs model bundle directory
  --pretrained               download models via the online feature
  --long-form                include the Silero VAD model
  --compute-units <mode>     cpu_only | cpu_and_gpu | all | cpu_and_neural_engine
  --show-ops <n>             print the first n ML Program operations"
    );
}

fn resolve_bundle(args: &Args) -> Result<ModelBundle> {
    if let Some(models_dir) = &args.models_dir {
        return Ok(ModelBundle::from_dir(models_dir));
    }

    #[cfg(feature = "online")]
    {
        #[cfg(feature = "long-form")]
        if args.long_form {
            return Ok(ModelBundle::from_pretrained_long_form()?);
        }

        #[cfg(not(feature = "long-form"))]
        if args.long_form {
            bail!("rebuild with --features long-form to inspect the VAD model")
        }

        let _ = args.pretrained;
        Ok(ModelBundle::from_pretrained()?)
    }

    #[cfg(not(feature = "online"))]
    {
        let _ = args.pretrained;
        bail!("rebuild with the default online feature or pass --models-dir")
    }
}

fn inspect_model(
    name: &str,
    path: &Path,
    compute_units: MLComputeUnits,
    show_ops: usize,
) -> Result<()> {
    if !path.exists() {
        bail!("missing model asset: {}", path.display())
    }

    println!("\n[{name}]");
    println!("path: {}", path.display());

    let compute_plan = load_compute_plan(path, compute_units)?;
    print_program_summary(&compute_plan, show_ops)?;
    Ok(())
}

fn load_compute_plan(
    path: &Path,
    compute_units: MLComputeUnits,
) -> Result<Retained<MLComputePlan>> {
    autoreleasepool(|_| {
        let path_str = NSString::from_str(&path.to_string_lossy());
        let url = NSURL::fileURLWithPath_isDirectory(&path_str, true);
        let config = unsafe { MLModelConfiguration::new() };
        unsafe { config.setComputeUnits(compute_units) };

        let (tx, rx) = mpsc::sync_channel(1);
        let handler = RcBlock::new(move |plan: *mut MLComputePlan, error: *mut NSError| {
            let result = if let Some(plan) = unsafe { Retained::retain(plan) } {
                Ok(plan)
            } else {
                let error = unsafe { Retained::retain(error) }
                    .map(|error| error.to_string())
                    .unwrap_or_else(|| "CoreML returned no compute plan and no error".to_owned());
                Err(error)
            };
            let _ = tx.send(result);
        });

        unsafe {
            MLComputePlan::loadContentsOfURL_configuration_completionHandler(
                &url, &config, &handler,
            )
        };

        rx.recv()
            .map_err(|error| eyre!("failed to receive compute plan: {error}"))?
            .map_err(|error| eyre!("failed to load compute plan: {error}"))
    })
}

fn print_program_summary(compute_plan: &MLComputePlan, show_ops: usize) -> Result<()> {
    let structure = unsafe { compute_plan.modelStructure() };
    let Some(program) = (unsafe { structure.program() }) else {
        let structure_kind = if unsafe { structure.neuralNetwork() }.is_some() {
            "neural_network"
        } else if unsafe { structure.pipeline() }.is_some() {
            "pipeline"
        } else {
            "other"
        };
        println!("structure: {structure_kind}");
        println!("compute_plan: no ML Program operations available");
        return Ok(());
    };

    println!("structure: ml_program");
    let main_function = main_function(&program)?;
    let block = unsafe { main_function.block() };
    let operations = unsafe { block.operations() };
    let total_ops = operations.count() as usize;
    println!("operations: {total_ops}");

    let mut preferred_counts = BTreeMap::new();
    let mut supported_devices = BTreeSet::new();

    for index in 0..total_ops {
        let operation = operations.objectAtIndex(index as _);
        let usage = unsafe { compute_plan.computeDeviceUsageForMLProgramOperation(&operation) };
        let operator_name = unsafe { operation.operatorName() }.to_string();

        let Some(usage) = usage else {
            if index < show_ops {
                println!("{index:04} {operator_name} preferred=unknown supported=unknown");
            }
            continue;
        };

        let preferred = describe_device(unsafe { usage.preferredComputeDevice() }.as_ref());
        *preferred_counts.entry(preferred.clone()).or_insert(0usize) += 1;

        let supported = describe_supported_devices(&usage);
        supported_devices.extend(supported.iter().cloned());

        if index < show_ops {
            println!(
                "{index:04} {operator_name} preferred={preferred} supported={}",
                supported.join(",")
            );
        }
    }

    println!("preferred_devices: {}", join_counts(&preferred_counts));
    println!("supported_devices: {}", join_items(&supported_devices));

    Ok(())
}

fn main_function(
    program: &MLModelStructureProgram,
) -> Result<Retained<MLModelStructureProgramFunction>> {
    let functions: Retained<NSDictionary<NSString, MLModelStructureProgramFunction>> =
        unsafe { program.functions() };
    let main_key = NSString::from_str("main");
    functions
        .objectForKey(&main_key)
        .ok_or_else(|| eyre!("ML Program was missing a `main` function"))
}

fn print_device_list(
    label: &str,
    devices: Retained<NSArray<ProtocolObject<dyn MLComputeDeviceProtocol>>>,
) {
    let mut names = Vec::new();
    for index in 0..devices.count() {
        let device = devices.objectAtIndex(index);
        names.push(describe_device(device.as_ref()));
    }
    println!("{label}: {}", names.join(", "));
}

fn describe_supported_devices(usage: &MLComputePlanDeviceUsage) -> Vec<String> {
    let devices = unsafe { usage.supportedComputeDevices() };
    let mut labels = Vec::with_capacity(devices.count());
    for index in 0..devices.count() {
        let device = devices.objectAtIndex(index);
        labels.push(describe_device(device.as_ref()));
    }
    labels.sort();
    labels.dedup();
    labels
}

fn describe_device(device: &ProtocolObject<dyn MLComputeDeviceProtocol>) -> String {
    let object: &AnyObject = device.as_ref();

    if object.downcast_ref::<MLCPUComputeDevice>().is_some() {
        return "CPU".to_owned();
    }

    if object.downcast_ref::<MLGPUComputeDevice>().is_some() {
        return "GPU".to_owned();
    }

    if let Some(device) = object.downcast_ref::<MLNeuralEngineComputeDevice>() {
        return format!("ANE({} cores)", unsafe { device.totalCoreCount() });
    }

    format!("{}", NSString::from_class(object.class()))
}

fn parse_compute_units(value: &str) -> Result<MLComputeUnits> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "cpu_only" => Ok(MLComputeUnits::CPUOnly),
        "cpu_and_gpu" => Ok(MLComputeUnits::CPUAndGPU),
        "all" => Ok(MLComputeUnits::All),
        "cpu_and_neural_engine" | "cpu_and_ane" | "default" => {
            Ok(MLComputeUnits::CPUAndNeuralEngine)
        }
        _ => bail!(
            "unsupported compute unit mode `{value}` expected cpu_only, cpu_and_gpu, all, or cpu_and_neural_engine"
        ),
    }
}

fn compute_units_name(value: MLComputeUnits) -> &'static str {
    match value {
        MLComputeUnits::CPUOnly => "cpu_only",
        MLComputeUnits::CPUAndGPU => "cpu_and_gpu",
        MLComputeUnits::All => "all",
        MLComputeUnits::CPUAndNeuralEngine => "cpu_and_neural_engine",
        _ => "unknown",
    }
}

fn next_path(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(next_value(args, flag)?))
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next().ok_or_else(|| eyre!("missing value for {flag}"))
}

fn join_counts(values: &BTreeMap<String, usize>) -> String {
    values
        .iter()
        .map(|(name, count)| format!("{name}={count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn join_items(values: &BTreeSet<String>) -> String {
    values.iter().cloned().collect::<Vec<_>>().join(", ")
}
