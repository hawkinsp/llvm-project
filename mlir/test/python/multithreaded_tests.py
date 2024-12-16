"""
This script generates multi-threaded tests to check free-threading mode using CPython compiled with TSAN.
Tests can be run using pytest:
```bash
python3.13t -mpytest -vvv multithreaded_tests.py
```

IMPORTANT. Running tests are not checking the correctness, but just the execution of the tests in multi-threaded context
and passing if no warnings reported by TSAN and failing otherwise.


Details on the generated tests and execution:
1) Multi-threaded execution: all generated tests are executed independently by
a pool of threads, running each test multiple times, see @multi_threaded for details

2) Tests generation: we use existing tests: test/python/ir/*.py,
test/python/dialects/*.py, etc to generate multi-threaded tests.
In details, we perform the following:
a) we define a list of source tests to be used to generate multi-threaded tests, see `TEST_MODULES`.
b) we define `TestAllMultiThreaded` class and add existing tests to the class. See `add_existing_tests` method.
c) for each test file, we copy and modify it: test/python/ir/affine_expr.py -> /tmp/ir/affine_expr.py.
In order to import the test file as python module, we remove all executing functions, like
`@run` or `run(testMethod)`. See `copy_and_update` and `add_existing_tests` methods for details.


Observed warnings reported by TSAN.
- Python 3.13.1 experimental free-threading build (tags/v3.13.1:06714517797, Dec 10 2024, 00:18:06) [Clang 15.0.7 ]
- numpy      2.3.0.dev0
- nanobind   2.5.0.dev1
- pybind11   master

CPython and free-threading known data-races:
1) ctypes  => try to build libffi with TSAN. Reported data races may be false positives. => can't build libffi from source
```
WARNING: ThreadSanitizer: data race (pid=99593)
  Atomic read of size 1 at 0x7f6054c485a8 by thread T3:
    #0 pthread_mutex_lock <null> (python3.13t+0xe83ca) (BuildId: de51a96f802ffcb0f2dcf5c04836201f1a81133c)
    #1 ffi_closure_alloc <null> (libffi.so.8+0x5d55) (BuildId: 59c2a6b204f74f358ca7711d2dfd349d88711f6a)
    #2 PyCFuncPtr_new /tmp/cpython-tsan/./Modules/_ctypes/_ctypes.c:3949:13 (_ctypes.cpython-313t-x86_64-linux-gnu.so+0x19205) (BuildId: 3866c4f0cc959b64602a68236b872ff98967ec7a)
    #3 type_call /tmp/cpython-tsan/Objects/typeobject.c:1981:11 (python3.13t+0x2d9080) (BuildId: de51a96f802ffcb0f2dcf5c04836201f1a81133c)
    #4 _PyObject_MakeTpCall /tmp/cpython-tsan/Objects/call.c:242:18 (python3.13t+0x1d6a6c) (BuildId: de51a96f802ffcb0f2dcf5c04836201f1a81133c)
```

LLVM related data-races
1) mlir pass manager
```
E               WARNING: ThreadSanitizer: data race (pid=109173)
E                 Write of size 8 at 0x7fef4f12a4a0 by thread T108 (mutexes: write M0):
E                   #0 llvm::raw_fd_ostream::write_impl(char const*, unsigned long) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:743:7 (libMLIRPythonCAPI.so.20.0git+0x4897686) (BuildId: 85b3b16da1be79a4)
E                   #1 llvm::raw_ostream::write(char const*, unsigned long) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:252:9 (libMLIRPythonCAPI.so.20.0git+0x4895aac) (BuildId: 85b3b16da1be79a4)
E                   #2 llvm::raw_ostream::operator<<(llvm::StringRef) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:230:14 (libMLIRPythonCAPI.so.20.0git+0x478ccbe) (BuildId: 85b3b16da1be79a4)
E                   #3 llvm::raw_ostream::operator<<(char const*) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:257:18 (libMLIRPythonCAPI.so.20.0git+0x478ccbe)
E                   #4 (anonymous namespace)::IRPrinterInstrumentation::runAfterPass(mlir::Pass*, mlir::Operation*)::$_1::operator()(llvm::raw_ostream&) const /tmp/jax/llvm-project/mlir/lib/Pass/IRPrinting.cpp:109:9 (libMLIRPythonCAPI.so.20.0git+0x478ccbe)
E                   #5 void llvm::function_ref<void (llvm::raw_ostream&)>::callback_fn<(anonymous namespace)::IRPrinterInstrumentation::runAfterPass(mlir::Pass*, mlir::Operation*)::$_1>(long, llvm::raw_ostream&) /tmp/jax/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:46:12 (libMLIRPythonCAPI.so.20.0git+0x478ccbe)
E                   #6 llvm::function_ref<void (llvm::raw_ostream&)>::operator()(llvm::raw_ostream&) const /tmp/jax/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:69:12 (libMLIRPythonCAPI.so.20.0git+0x478d54e) (BuildId: 85b3b16da1be79a4)
E                   #7 (anonymous namespace)::BasicIRPrinterConfig::printAfterIfEnabled(mlir::Pass*, mlir::Operation*, llvm::function_ref<void (llvm::raw_ostream&)>) /tmp/jax/llvm-project/mlir/lib/Pass/IRPrinting.cpp:195:7 (libMLIRPythonCAPI.so.20.0git+0x478d54e)
E                   #8 (anonymous namespace)::IRPrinterInstrumentation::runAfterPass(mlir::Pass*, mlir::Operation*) /tmp/jax/llvm-project/mlir/lib/Pass/IRPrinting.cpp:108:11 (libMLIRPythonCAPI.so.20.0git+0x478b7d2) (BuildId: 85b3b16da1be79a4)
E                   #9 mlir::PassInstrumentor::runAfterPass(mlir::Pass*, mlir::Operation*) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:1037:12 (libMLIRPythonCAPI.so.20.0git+0x4797282) (BuildId: 85b3b16da1be79a4)
E                   #10 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:563:11 (libMLIRPythonCAPI.so.20.0git+0x4797282)
E                   #11 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:592:16 (libMLIRPythonCAPI.so.20.0git+0x479b55b) (BuildId: 85b3b16da1be79a4)
E                   #12 mlir::PassManager::runPasses(mlir::Operation*, mlir::AnalysisManager) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:905:10 (libMLIRPythonCAPI.so.20.0git+0x479b55b)
E                   #13 mlir::PassManager::run(mlir::Operation*) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:885:60 (libMLIRPythonCAPI.so.20.0git+0x479b55b)
E                   #14 mlirPassManagerRunOnOp /tmp/jax/llvm-project/mlir/lib/CAPI/IR/Pass.cpp:44:36 (libMLIRPythonCAPI.so.20.0git+0x46c6150) (BuildId: 85b3b16da1be79a4)
E                   #15 mlir::python::populatePassManagerSubmodule(pybind11::module_&)::$_5::operator()((anonymous namespace)::PyPassManager&, mlir::python::PyOperationBase&, bool) const /tmp/jax/llvm-project/mlir/lib/Bindings/Python/Pass.cpp:154:40 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x343e3e) (BuildId: 6e516b7f12acec76)
E                   #16 void pybind11::detail::argument_loader<(anonymous namespace)::PyPassManager&, mlir::python::PyOperationBase&, bool>::call_impl<void, mlir::python::populatePassManagerSubmodule(pybind11::module_&)::$_5&, 0ul, 1ul, 2ul, pybind11::detail::void_type>(mlir::python::populatePassManagerSubmodule(pybind11::module_&)::$_5&, std::integer_sequence<unsigned long, 0ul, 1ul, 2ul>, pybind11::detail::void_type&&) && /usr/local/include/pybind11/cast.h:1685:16 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x343e3e)
```

```
E               WARNING: ThreadSanitizer: data race (pid=13402)
E                 Write of size 8 at 0x7f5bf342a4a0 by thread T63:
E                   #0 llvm::raw_fd_ostream::write_impl(char const*, unsigned long) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:743:7 (libMLIRPythonCAPI.so.20.0git+0x4897686) (BuildId: 85b3b16da1be79a4)
E                   #1 llvm::raw_ostream::write(char const*, unsigned long) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:252:9 (libMLIRPythonCAPI.so.20.0git+0x4895aac) (BuildId: 85b3b16da1be79a4)
E                   #2 llvm::raw_ostream::operator<<(llvm::StringRef) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:230:14 (libMLIRPythonCAPI.so.20.0git+0x6996bd2) (BuildId: 85b3b16da1be79a4)
E                   #3 llvm::raw_ostream::operator<<(char const*) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:257:18 (libMLIRPythonCAPI.so.20.0git+0x6996bd2)
E                   #4 (anonymous namespace)::PrintOpStatsPass::printSummary() /tmp/jax/llvm-project/mlir/lib/Transforms/OpStats.cpp:62:6 (libMLIRPythonCAPI.so.20.0git+0x6996bd2)
E                   #5 (anonymous namespace)::PrintOpStatsPass::runOnOperation() /tmp/jax/llvm-project/mlir/lib/Transforms/OpStats.cpp:57:5 (libMLIRPythonCAPI.so.20.0git+0x6996bd2)
E                   #6 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_7::operator()() const /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:526:17 (libMLIRPythonCAPI.so.20.0git+0x4796dc4) (BuildId: 85b3b16da1be79a4)
E                   #7 void llvm::function_ref<void ()>::callback_fn<mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_7>(long) /tmp/jax/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:46:12 (libMLIRPythonCAPI.so.20.0git+0x4796dc4)
E                   #8 llvm::function_ref<void ()>::operator()() const /tmp/jax/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:69:12 (libMLIRPythonCAPI.so.20.0git+0x4796dc4)
E                   #9 void mlir::MLIRContext::executeAction<mlir::PassExecutionAction, mlir::Pass&>(llvm::function_ref<void ()>, llvm::ArrayRef<mlir::IRUnit>, mlir::Pass&) /tmp/jax/llvm-project/mlir/include/mlir/IR/MLIRContext.h:280:7 (libMLIRPythonCAPI.so.20.0git+0x4796dc4)
E                   #10 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:520:21 (libMLIRPythonCAPI.so.20.0git+0x4796dc4)
E                   #11 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:592:16 (libMLIRPythonCAPI.so.20.0git+0x479b55b) (BuildId: 85b3b16da1be79a4)
E                   #12 mlir::PassManager::runPasses(mlir::Operation*, mlir::AnalysisManager) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:905:10 (libMLIRPythonCAPI.so.20.0git+0x479b55b)
E                   #13 mlir::PassManager::run(mlir::Operation*) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:885:60 (libMLIRPythonCAPI.so.20.0git+0x479b55b)
E                   #14 mlirPassManagerRunOnOp /tmp/jax/llvm-project/mlir/lib/CAPI/IR/Pass.cpp:44:36 (libMLIRPythonCAPI.so.20.0git+0x46c6150) (BuildId: 85b3b16da1be79a4)
E                   #15 mlir::python::populatePassManagerSubmodule(pybind11::module_&)::$_5::operator()((anonymous namespace)::PyPassManager&, mlir::python::PyOperationBase&, bool) const /tmp/jax/llvm-project/mlir/lib/Bindings/Python/Pass.cpp:154:40 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x343e3e) (BuildId: 6e516b7f12acec76)
E                   #16 void pybind11::detail::argument_loader<(anonymous namespace)::PyPassManager&, mlir::python::PyOperationBase&, bool>::call_impl<void, mlir::python::populatePassManagerSubmodule(pybind11::module_&)::$_5&, 0ul, 1ul, 2ul, pybind11::detail::void_type>(mlir::python::populatePassManagerSubmodule(pybind11::module_&)::$_5&, std::integer_sequence<unsigned long, 0ul, 1ul, 2ul>, pybind11::detail::void_type&&) && /usr/local/include/pybind11/cast.h:1685:16 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x343e3e)
```


```
E               WARNING: ThreadSanitizer: data race (pid=14122)
E                 Write of size 8 at 0x7fa88142a4a0 by thread T59 (mutexes: write M0):
E                   #0 llvm::raw_fd_ostream::write_impl(char const*, unsigned long) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:743:7 (libMLIRPythonCAPI.so.20.0git+0x4897686) (BuildId: 85b3b16da1be79a4)
E                   #1 llvm::raw_ostream::write(char const*, unsigned long) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:252:9 (libMLIRPythonCAPI.so.20.0git+0x4895aac) (BuildId: 85b3b16da1be79a4)
E                   #2 llvm::raw_ostream::operator<<(llvm::StringRef) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:230:14 (libMLIRPythonCAPI.so.20.0git+0x478c5be) (BuildId: 85b3b16da1be79a4)
E                   #3 llvm::raw_ostream::operator<<(char const*) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:257:18 (libMLIRPythonCAPI.so.20.0git+0x478c5be)
E                   #4 (anonymous namespace)::IRPrinterInstrumentation::runBeforePass(mlir::Pass*, mlir::Operation*)::$_0::operator()(llvm::raw_ostream&) const /tmp/jax/llvm-project/mlir/lib/Pass/IRPrinting.cpp:78:9 (libMLIRPythonCAPI.so.20.0git+0x478c5be)
E                   #5 void llvm::function_ref<void (llvm::raw_ostream&)>::callback_fn<(anonymous namespace)::IRPrinterInstrumentation::runBeforePass(mlir::Pass*, mlir::Operation*)::$_0>(long, llvm::raw_ostream&) /tmp/jax/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:46:12 (libMLIRPythonCAPI.so.20.0git+0x478c5be)
E                   #6 llvm::function_ref<void (llvm::raw_ostream&)>::operator()(llvm::raw_ostream&) const /tmp/jax/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:69:12 (libMLIRPythonCAPI.so.20.0git+0x478d49e) (BuildId: 85b3b16da1be79a4)
E                   #7 (anonymous namespace)::BasicIRPrinterConfig::printBeforeIfEnabled(mlir::Pass*, mlir::Operation*, llvm::function_ref<void (llvm::raw_ostream&)>) /tmp/jax/llvm-project/mlir/lib/Pass/IRPrinting.cpp:189:7 (libMLIRPythonCAPI.so.20.0git+0x478d49e)
E                   #8 (anonymous namespace)::IRPrinterInstrumentation::runBeforePass(mlir::Pass*, mlir::Operation*) /tmp/jax/llvm-project/mlir/lib/Pass/IRPrinting.cpp:77:11 (libMLIRPythonCAPI.so.20.0git+0x478b506) (BuildId: 85b3b16da1be79a4)
E                   #9 mlir::PassInstrumentor::runBeforePass(mlir::Pass*, mlir::Operation*) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:1030:12 (libMLIRPythonCAPI.so.20.0git+0x4796ca2) (BuildId: 85b3b16da1be79a4)
E                   #10 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:517:9 (libMLIRPythonCAPI.so.20.0git+0x4796ca2)
E                   #11 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:592:16 (libMLIRPythonCAPI.so.20.0git+0x479b55b) (BuildId: 85b3b16da1be79a4)
E                   #12 mlir::PassManager::runPasses(mlir::Operation*, mlir::AnalysisManager) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:905:10 (libMLIRPythonCAPI.so.20.0git+0x479b55b)
E                   #13 mlir::PassManager::run(mlir::Operation*) /tmp/jax/llvm-project/mlir/lib/Pass/Pass.cpp:885:60 (libMLIRPythonCAPI.so.20.0git+0x479b55b)
E                   #14 mlirPassManagerRunOnOp /tmp/jax/llvm-project/mlir/lib/CAPI/IR/Pass.cpp:44:36 (libMLIRPythonCAPI.so.20.0git+0x46c6150) (BuildId: 85b3b16da1be79a4)
E                   #15 mlir::python::populatePassManagerSubmodule(pybind11::module_&)::$_5::operator()((anonymous namespace)::PyPassManager&, mlir::python::PyOperationBase&, bool) const /tmp/jax/llvm-project/mlir/lib/Bindings/Python/Pass.cpp:154:40 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x343e3e) (BuildId: 6e516b7f12acec76)
E                   #16 void pybind11::detail::argument_loader<(anonymous namespace)::PyPassManager&, mlir::python::PyOperationBase&, bool>::call_impl<void, mlir::python::populatePassManagerSubmodule(pybind11::module_&)::$_5&, 0ul, 1ul, 2ul, pybind11::detail::void_type>(mlir::python::populatePassManagerSubmodule(pybind11::module_&)::$_5&, std::integer_sequence<unsigned long, 0ul, 1ul, 2ul>, pybind11::detail::void_type&&) && /usr/local/include/pybind11/cast.h:1685:16 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x343e3e)
```


2) dialects/transform_interpreter.py

```
E               WARNING: ThreadSanitizer: data race (pid=15594)
E                 Read of size 8 at 0x7fdece62a3f8 by thread T58:
E                   #0 llvm::raw_ostream::operator<<(llvm::StringRef) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:229:25 (libMLIRPythonCAPI.so.20.0git+0x5963297) (BuildId: 85b3b16da1be79a4)
E                   #1 llvm::raw_ostream::operator<<(char const*) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:257:18 (libMLIRPythonCAPI.so.20.0git+0x5963297)
E                   #2 mlir::transform::PrintOp::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/IR/TransformOps.cpp:2787:16 (libMLIRPythonCAPI.so.20.0git+0x5963297)
E                   #3 mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Model<mlir::transform::PrintOp>::apply(mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Concept const*, mlir::Operation*, mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h.inc:477:56 (libMLIRPythonCAPI.so.20.0git+0x58e1e9a) (BuildId: 85b3b16da1be79a4)
E                   #4 mlir::transform::TransformOpInterface::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.cpp.inc:61:14 (libMLIRPythonCAPI.so.20.0git+0xb246960) (BuildId: 85b3b16da1be79a4)
E                   #5 mlir::transform::TransformState::applyTransform(mlir::transform::TransformOpInterface) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Interfaces/TransformInterfaces.cpp:951:48 (libMLIRPythonCAPI.so.20.0git+0xb246960)
E                   #6 applySequenceBlock(mlir::Block&, mlir::transform::FailurePropagationMode, mlir::transform::TransformState&, mlir::transform::TransformResults&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/IR/TransformOps.cpp:1786:15 (libMLIRPythonCAPI.so.20.0git+0x5958aa7) (BuildId: 85b3b16da1be79a4)
E                   #7 mlir::transform::IncludeOp::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/IR/TransformOps.cpp:1827:40 (libMLIRPythonCAPI.so.20.0git+0x59584b3) (BuildId: 85b3b16da1be79a4)
E                   #8 mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Model<mlir::transform::IncludeOp>::apply(mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Concept const*, mlir::Operation*, mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h.inc:477:56 (libMLIRPythonCAPI.so.20.0git+0x58d4dfa) (BuildId: 85b3b16da1be79a4)
E                   #9 mlir::transform::TransformOpInterface::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.cpp.inc:61:14 (libMLIRPythonCAPI.so.20.0git+0xb246960) (BuildId: 85b3b16da1be79a4)
E                   #10 mlir::transform::TransformState::applyTransform(mlir::transform::TransformOpInterface) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Interfaces/TransformInterfaces.cpp:951:48 (libMLIRPythonCAPI.so.20.0git+0xb246960)
E                   #11 applySequenceBlock(mlir::Block&, mlir::transform::FailurePropagationMode, mlir::transform::TransformState&, mlir::transform::TransformResults&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/IR/TransformOps.cpp:1786:15 (libMLIRPythonCAPI.so.20.0git+0x5958aa7) (BuildId: 85b3b16da1be79a4)
E                   #12 mlir::transform::IncludeOp::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/IR/TransformOps.cpp:1827:40 (libMLIRPythonCAPI.so.20.0git+0x59584b3) (BuildId: 85b3b16da1be79a4)
E                   #13 mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Model<mlir::transform::IncludeOp>::apply(mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Concept const*, mlir::Operation*, mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h.inc:477:56 (libMLIRPythonCAPI.so.20.0git+0x58d4dfa) (BuildId: 85b3b16da1be79a4)
E                   #14 mlir::transform::TransformOpInterface::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.cpp.inc:61:14 (libMLIRPythonCAPI.so.20.0git+0xb246960) (BuildId: 85b3b16da1be79a4)
E                   #15 mlir::transform::TransformState::applyTransform(mlir::transform::TransformOpInterface) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Interfaces/TransformInterfaces.cpp:951:48 (libMLIRPythonCAPI.so.20.0git+0xb246960)
E                   #16 applySequenceBlock(mlir::Block&, mlir::transform::FailurePropagationMode, mlir::transform::TransformState&, mlir::transform::TransformResults&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/IR/TransformOps.cpp:1786:15 (libMLIRPythonCAPI.so.20.0git+0x5958aa7) (BuildId: 85b3b16da1be79a4)
E                   #17 mlir::transform::NamedSequenceOp::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/IR/TransformOps.cpp:2153:10 (libMLIRPythonCAPI.so.20.0git+0x595ea87) (BuildId: 85b3b16da1be79a4)
E                   #18 mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Model<mlir::transform::NamedSequenceOp>::apply(mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Concept const*, mlir::Operation*, mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h.inc:477:56 (libMLIRPythonCAPI.so.20.0git+0x58dda7a) (BuildId: 85b3b16da1be79a4)
E                   #19 mlir::transform::TransformOpInterface::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.cpp.inc:61:14 (libMLIRPythonCAPI.so.20.0git+0xb246960) (BuildId: 85b3b16da1be79a4)
E                   #20 mlir::transform::TransformState::applyTransform(mlir::transform::TransformOpInterface) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Interfaces/TransformInterfaces.cpp:951:48 (libMLIRPythonCAPI.so.20.0git+0xb246960)
E                   #21 mlir::transform::applyTransforms(mlir::Operation*, mlir::transform::TransformOpInterface, mlir::RaggedArray<llvm::PointerUnion<mlir::Operation*, mlir::Attribute, mlir::Value>> const&, mlir::transform::TransformOptions const&, bool, llvm::function_ref<void (mlir::transform::TransformState&)>, llvm::function_ref<llvm::LogicalResult (mlir::transform::TransformState&)>) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Interfaces/TransformInterfaces.cpp:2018:13 (libMLIRPythonCAPI.so.20.0git+0xb256a9c) (BuildId: 85b3b16da1be79a4)
E                   #22 mlir::transform::applyTransformNamedSequence(mlir::RaggedArray<llvm::PointerUnion<mlir::Operation*, mlir::Attribute, mlir::Value>>, mlir::transform::TransformOpInterface, mlir::ModuleOp, mlir::transform::TransformOptions const&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Transforms/TransformInterpreterUtils.cpp:234:10 (libMLIRPythonCAPI.so.20.0git+0x59991ab) (BuildId: 85b3b16da1be79a4)
E                   #23 mlir::transform::applyTransformNamedSequence(mlir::Operation*, mlir::Operation*, mlir::ModuleOp, mlir::transform::TransformOptions const&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Transforms/TransformInterpreterUtils.cpp:196:10 (libMLIRPythonCAPI.so.20.0git+0x59986c4) (BuildId: 85b3b16da1be79a4)
E                   #24 mlirTransformApplyNamedSequence /tmp/jax/llvm-project/mlir/lib/CAPI/Dialect/TransformInterpreter.cpp:71:15 (libMLIRPythonCAPI.so.20.0git+0x46d16b4) (BuildId: 85b3b16da1be79a4)
E                   #25 populateTransformInterpreterSubmodule(pybind11::module_&)::$_4::operator()(MlirOperation, MlirOperation, MlirOperation, (anonymous namespace)::PyMlirTransformOptions const&) const /tmp/jax/llvm-project/mlir/lib/Bindings/Python/TransformInterpreter.cpp:74:36 (_mlirTransformInterpreter.cpython-313t-x86_64-linux-gnu.so+0x3fc0d) (BuildId: 7d3d2fbf5b54f7b5)
E                   #26 void pybind11::detail::argument_loader<MlirOperation, MlirOperation, MlirOperation, (anonymous namespace)::PyMlirTransformOptions const&>::call_impl<void, populateTransformInterpreterSubmodule(pybind11::module_&)::$_4&, 0ul, 1ul, 2ul, 3ul, pybind11::detail::void_type>(populateTransformInterpreterSubmodule(pybind11::module_&)::$_4&, std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul>, pybind11::detail::void_type&&) && /usr/local/include/pybind11/cast.h:1685:16 (_mlirTransformInterpreter.cpython-313t-x86_64-linux-gnu.so+0x3fc0d)
```

```
E               WARNING: ThreadSanitizer: data race (pid=16690)
E                 Read of size 8 at 0x7fcda1b2a3f8 by thread T62:
E                   #0 llvm::raw_ostream::operator<<(llvm::StringRef) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:229:25 (libMLIRPythonCAPI.so.20.0git+0x5963297) (BuildId: 85b3b16da1be79a4)
E                   #1 llvm::raw_ostream::operator<<(char const*) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:257:18 (libMLIRPythonCAPI.so.20.0git+0x5963297)
E                   #2 mlir::transform::PrintOp::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/IR/TransformOps.cpp:2787:16 (libMLIRPythonCAPI.so.20.0git+0x5963297)
E                   #3 mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Model<mlir::transform::PrintOp>::apply(mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Concept const*, mlir::Operation*, mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h.inc:477:56 (libMLIRPythonCAPI.so.20.0git+0x58e1e9a) (BuildId: 85b3b16da1be79a4)
E                   #4 mlir::transform::TransformOpInterface::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.cpp.inc:61:14 (libMLIRPythonCAPI.so.20.0git+0xb246960) (BuildId: 85b3b16da1be79a4)
E                   #5 mlir::transform::TransformState::applyTransform(mlir::transform::TransformOpInterface) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Interfaces/TransformInterfaces.cpp:951:48 (libMLIRPythonCAPI.so.20.0git+0xb246960)
E                   #6 applySequenceBlock(mlir::Block&, mlir::transform::FailurePropagationMode, mlir::transform::TransformState&, mlir::transform::TransformResults&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/IR/TransformOps.cpp:1786:15 (libMLIRPythonCAPI.so.20.0git+0x5958aa7) (BuildId: 85b3b16da1be79a4)
E                   #7 mlir::transform::NamedSequenceOp::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/IR/TransformOps.cpp:2153:10 (libMLIRPythonCAPI.so.20.0git+0x595ea87) (BuildId: 85b3b16da1be79a4)
E                   #8 mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Model<mlir::transform::NamedSequenceOp>::apply(mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Concept const*, mlir::Operation*, mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h.inc:477:56 (libMLIRPythonCAPI.so.20.0git+0x58dda7a) (BuildId: 85b3b16da1be79a4)
E                   #9 mlir::transform::TransformOpInterface::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) /tmp/jax/llvm-project/build-tsan/tools/mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.cpp.inc:61:14 (libMLIRPythonCAPI.so.20.0git+0xb246960) (BuildId: 85b3b16da1be79a4)
E                   #10 mlir::transform::TransformState::applyTransform(mlir::transform::TransformOpInterface) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Interfaces/TransformInterfaces.cpp:951:48 (libMLIRPythonCAPI.so.20.0git+0xb246960)
E                   #11 mlir::transform::applyTransforms(mlir::Operation*, mlir::transform::TransformOpInterface, mlir::RaggedArray<llvm::PointerUnion<mlir::Operation*, mlir::Attribute, mlir::Value>> const&, mlir::transform::TransformOptions const&, bool, llvm::function_ref<void (mlir::transform::TransformState&)>, llvm::function_ref<llvm::LogicalResult (mlir::transform::TransformState&)>) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Interfaces/TransformInterfaces.cpp:2018:13 (libMLIRPythonCAPI.so.20.0git+0xb256a9c) (BuildId: 85b3b16da1be79a4)
E                   #12 mlir::transform::applyTransformNamedSequence(mlir::RaggedArray<llvm::PointerUnion<mlir::Operation*, mlir::Attribute, mlir::Value>>, mlir::transform::TransformOpInterface, mlir::ModuleOp, mlir::transform::TransformOptions const&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Transforms/TransformInterpreterUtils.cpp:234:10 (libMLIRPythonCAPI.so.20.0git+0x59991ab) (BuildId: 85b3b16da1be79a4)
E                   #13 mlir::transform::applyTransformNamedSequence(mlir::Operation*, mlir::Operation*, mlir::ModuleOp, mlir::transform::TransformOptions const&) /tmp/jax/llvm-project/mlir/lib/Dialect/Transform/Transforms/TransformInterpreterUtils.cpp:196:10 (libMLIRPythonCAPI.so.20.0git+0x59986c4) (BuildId: 85b3b16da1be79a4)
E                   #14 mlirTransformApplyNamedSequence /tmp/jax/llvm-project/mlir/lib/CAPI/Dialect/TransformInterpreter.cpp:71:15 (libMLIRPythonCAPI.so.20.0git+0x46d16b4) (BuildId: 85b3b16da1be79a4)
E                   #15 populateTransformInterpreterSubmodule(pybind11::module_&)::$_4::operator()(MlirOperation, MlirOperation, MlirOperation, (anonymous namespace)::PyMlirTransformOptions const&) const /tmp/jax/llvm-project/mlir/lib/Bindings/Python/TransformInterpreter.cpp:74:36 (_mlirTransformInterpreter.cpython-313t-x86_64-linux-gnu.so+0x3fc0d) (BuildId: 7d3d2fbf5b54f7b5)
E                   #16 void pybind11::detail::argument_loader<MlirOperation, MlirOperation, MlirOperation, (anonymous namespace)::PyMlirTransformOptions const&>::call_impl<void, populateTransformInterpreterSubmodule(pybind11::module_&)::$_4&, 0ul, 1ul, 2ul, 3ul, pybind11::detail::void_type>(populateTransformInterpreterSubmodule(pybind11::module_&)::$_4&, std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul>, pybind11::detail::void_type&&) && /usr/local/include/pybind11/cast.h:1685:16 (_mlirTransformInterpreter.cpython-313t-x86_64-linux-gnu.so+0x3fc0d)
```


3) ir/diagnostic_handler.py
```
E               WARNING: ThreadSanitizer: data race (pid=19144)
E                 Write of size 8 at 0x7ff1be8294a0 by thread T60 (mutexes: write M0):
E                   #0 llvm::raw_fd_ostream::write_impl(char const*, unsigned long) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:743:7 (libMLIRPythonCAPI.so.20.0git+0x4897686) (BuildId: 85b3b16da1be79a4)
E                   #1 llvm::raw_ostream::write(char const*, unsigned long) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:252:9 (libMLIRPythonCAPI.so.20.0git+0x4895aac) (BuildId: 85b3b16da1be79a4)
E                   #2 llvm::raw_ostream::operator<<(llvm::StringRef) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:230:14 (libMLIRPythonCAPI.so.20.0git+0x49d130f) (BuildId: 85b3b16da1be79a4)
E                   #3 llvm::raw_ostream::operator<<(char const*) /tmp/jax/llvm-project/llvm/include/llvm/Support/raw_ostream.h:257:18 (libMLIRPythonCAPI.so.20.0git+0x49d130f)
E                   #4 mlir::detail::DiagnosticEngineImpl::emit(mlir::Diagnostic&&) /tmp/jax/llvm-project/mlir/lib/IR/Diagnostics.cpp:264:6 (libMLIRPythonCAPI.so.20.0git+0x49d130f)
E                   #5 mlir::DiagnosticEngine::emit(mlir::Diagnostic&&) /tmp/jax/llvm-project/mlir/lib/IR/Diagnostics.cpp:299:9 (libMLIRPythonCAPI.so.20.0git+0x49d0ff4) (BuildId: 85b3b16da1be79a4)
E                   #6 mlir::InFlightDiagnostic::report() /tmp/jax/llvm-project/mlir/lib/IR/Diagnostics.cpp:210:12 (libMLIRPythonCAPI.so.20.0git+0x49d0ff4)
E                   #7 mlir::InFlightDiagnostic::~InFlightDiagnostic() /tmp/jax/llvm-project/mlir/include/mlir/IR/Diagnostics.h:325:7 (libMLIRPythonCAPI.so.20.0git+0x46bae43) (BuildId: 85b3b16da1be79a4)
E                   #8 mlirEmitError /tmp/jax/llvm-project/mlir/lib/CAPI/IR/Diagnostics.cpp:79:3 (libMLIRPythonCAPI.so.20.0git+0x46bae43)
E                   #9 mlir::python::populateIRCore(pybind11::module_&)::$_43::operator()(mlir::python::PyLocation&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>) const /tmp/jax/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp:2939:13 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x26cce9) (BuildId: d60a319200e3423d)
E                   #10 void pybind11::detail::argument_loader<mlir::python::PyLocation&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>>::call_impl<void, mlir::python::populateIRCore(pybind11::module_&)::$_43&, 0ul, 1ul, pybind11::detail::void_type>(mlir::python::populateIRCore(pybind11::module_&)::$_43&, std::integer_sequence<unsigned long, 0ul, 1ul>, pybind11::detail::void_type&&) && /usr/local/include/pybind11/cast.h:1685:16 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x26cce9)
```

5) ir/module.py
```
E               WARNING: ThreadSanitizer: data race (pid=19407)
E                 Write of size 8 at 0x7f1858429468 by thread T62:
E                   #0 llvm::raw_ostream::copy_to_buffer(char const*, unsigned long) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:306:13 (libMLIRPythonCAPI.so.20.0git+0x48958bb) (BuildId: 85b3b16da1be79a4)
E                   #1 llvm::raw_ostream::write(char const*, unsigned long) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:285:3 (libMLIRPythonCAPI.so.20.0git+0x48958bb)
E                   #2 llvm::raw_ostream& write_padding<(char)32>(llvm::raw_ostream&, unsigned int) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:486:15 (libMLIRPythonCAPI.so.20.0git+0x489610e) (BuildId: 85b3b16da1be79a4)
E                   #3 llvm::raw_ostream::indent(unsigned int) /tmp/jax/llvm-project/llvm/lib/Support/raw_ostream.cpp:498:10 (libMLIRPythonCAPI.so.20.0git+0x489610e)
E                   #4 (anonymous namespace)::OperationPrinter::printFullOpWithIndentAndLoc(mlir::Operation*) /tmp/jax/llvm-project/mlir/lib/IR/AsmPrinter.cpp:3426:6 (libMLIRPythonCAPI.so.20.0git+0x4912af0) (BuildId: 85b3b16da1be79a4)
E                   #5 mlir::Operation::print(llvm::raw_ostream&, mlir::AsmState&) /tmp/jax/llvm-project/mlir/lib/IR/AsmPrinter.cpp:3979:13 (libMLIRPythonCAPI.so.20.0git+0x4910238) (BuildId: 85b3b16da1be79a4)
E                   #6 mlir::Operation::print(llvm::raw_ostream&, mlir::OpPrintingFlags const&) /tmp/jax/llvm-project/mlir/lib/IR/AsmPrinter.cpp:3971:3 (libMLIRPythonCAPI.so.20.0git+0x4913a05) (BuildId: 85b3b16da1be79a4)
E                   #7 mlir::Operation::dump() /tmp/jax/llvm-project/mlir/lib/IR/AsmPrinter.cpp:3984:3 (libMLIRPythonCAPI.so.20.0git+0x4913a05)
E                   #8 mlirOperationDump /tmp/jax/llvm-project/mlir/lib/CAPI/IR/IR.cpp:715:63 (libMLIRPythonCAPI.so.20.0git+0x46c05d9) (BuildId: 85b3b16da1be79a4)
E                   #9 mlir::python::populateIRCore(pybind11::module_&)::$_50::operator()(mlir::python::PyModule&) const /tmp/jax/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp:3000:13 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x26fc90) (BuildId: d60a319200e3423d)
E                   #10 void pybind11::detail::argument_loader<mlir::python::PyModule&>::call_impl<void, mlir::python::populateIRCore(pybind11::module_&)::$_50&, 0ul, pybind11::detail::void_type>(mlir::python::populateIRCore(pybind11::module_&)::$_50&, std::integer_sequence<unsigned long, 0ul>, pybind11::detail::void_type&&) && /usr/local/include/pybind11/cast.h:1685:16 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x26fc90)
```

```
E               WARNING: ThreadSanitizer: data race (pid=19669)
E                 Write of size 8 at 0x7f3882591840 by thread T62:
E                   #0 unicode_fill_utf8 /tmp/cpython-tsan/Objects/unicodeobject.c:5444:30 (python3.13t+0x3142a1) (BuildId: 31ded8fac9cea6b44dd4f3246be3853edec8d0dc)
E                   #1 PyUnicode_AsUTF8AndSize /tmp/cpython-tsan/Objects/unicodeobject.c:4066:13 (python3.13t+0x3142a1)
E                   #2 pybind11::detail::string_caster<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>, false>::load(pybind11::handle, bool) /usr/local/include/pybind11/cast.h:444:51 (_mlir.cpython-313t-x86_64-linux-gnu.so+0xd6a25) (BuildId: d60a319200e3423d)
E                   #3 bool pybind11::detail::argument_loader<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&, mlir::python::DefaultingPyMlirContext>::load_impl_sequence<0ul, 1ul>(pybind11::detail::function_call&, std::integer_sequence<unsigned long, 0ul, 1ul>) /usr/local/include/pybind11/cast.h:1670:47 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x26e4a5) (BuildId: d60a319200e3423d)
E                   #4 pybind11::detail::argument_loader<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&, mlir::python::DefaultingPyMlirContext>::load_args(pybind11::detail::function_call&) /usr/local/include/pybind11/cast.h:1648:50 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x26e036) (BuildId: d60a319200e3423d)
E                   #5 void pybind11::cpp_function::initialize<mlir::python::populateIRCore(pybind11::module_&)::$_45, pybind11::object, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&, mlir::python::DefaultingPyMlirContext, pybind11::name, pybind11::scope, pybind11::sibling, pybind11::arg, pybind11::arg_v, char [168]>(mlir::python::populateIRCore(pybind11::module_&)::$_45&&, pybind11::object (*)(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&, mlir::python::DefaultingPyMlirContext), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&, pybind11::arg const&, pybind11::arg_v const&, char const (&) [168])::'lambda'(pybind11::detail::function_call&)::operator()(pybind11::detail::function_call&) const /usr/local/include/pybind11/pybind11.h:259:33 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x26e036)
E                   #6 void pybind11::cpp_function::initialize<mlir::python::populateIRCore(pybind11::module_&)::$_45, pybind11::object, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&, mlir::python::DefaultingPyMlirContext, pybind11::name, pybind11::scope, pybind11::sibling, pybind11::arg, pybind11::arg_v, char [168]>(mlir::python::populateIRCore(pybind11::module_&)::$_45&&, pybind11::object (*)(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&, mlir::python::DefaultingPyMlirContext), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&, pybind11::arg const&, pybind11::arg_v const&, char const (&) [168])::'lambda'(pybind11::detail::function_call&)::__invoke(pybind11::detail::function_call&) /usr/local/include/pybind11/pybind11.h:255:21 (_mlir.cpython-313t-x86_64-linux-gnu.so+0x26e036)
E                   #7 pybind11::cpp_function::dispatcher(_object*, _object*, _object*) /usr/local/include/pybind11/pybind11.h:986:30 (_mlir.cpython-313t-x86_64-linux-gnu.so+0xe6227) (BuildId: d60a319200e3423d)
E                   #8 cfunction_call /tmp/cpython-tsan/Objects/methodobject.c:540:18 (python3.13t+0x2795e7) (BuildId: 31ded8fac9cea6b44dd4f3246be3853edec8d0dc)

```

"""
import concurrent.futures
import functools
import gc
import importlib.util
import sys
import threading
import tempfile

from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional

import pytest

import mlir.dialects.arith as arith
from mlir.dialects import transform
from mlir.ir import Context, Location, Module, IntegerType, F64Type, InsertionPoint


def import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def copy_and_update(src_filepath: Path, dst_filepath: Path):
    # We should remove all calls like `run(testMethod)`
    with open(src_filepath, "r") as reader, open(dst_filepath, "w") as writer:
        while True:
            src_line = reader.readline()
            if len(src_line) == 0:
                break
            skip_lines = [
                "run(",
                "@run",
                "@constructAndPrintInModule",
                "run_apply_patterns(",
                "@run_apply_patterns",
                "@test_in_context",
                "@construct_and_print_in_module",
            ]
            if any(src_line.startswith(line) for line in skip_lines):
                continue
            writer.write(src_line)


# Helper run functions
def run(f):
    f()


def run_with_context_and_location(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


def run_with_insertion_point(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f(ctx)
        print(module)


def run_with_insertion_point_v2(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


def run_with_insertion_point_v3(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f(module)
        print(module)
    return f


def run_with_insertion_point_v4(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = Module.create()
        with InsertionPoint(module.body):
            f()
    return f


def run_apply_patterns(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            sequence = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                [],
                transform.AnyOpType.get(),
            )
            with InsertionPoint(sequence.body):
                apply = transform.ApplyPatternsOp(sequence.bodyTarget)
                with InsertionPoint(apply.patterns):
                    f()
                transform.YieldOp()
        print("\nTEST:", f.__name__)
        print(module)
    return f


def run_transform_tensor_ext(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            sequence = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                [],
                transform.AnyOpType.get(),
            )
            with InsertionPoint(sequence.body):
                f(sequence.bodyTarget)
                transform.YieldOp()
        print(module)
    return f


def run_transform_structured_ext(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        module.operation.verify()
        print(module)
    return f


def run_construct_and_print_in_module(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            module = f(module)
        if module is not None:
            print(module)
    return f


TEST_MODULES = [
    ("execution_engine", run),  # Fail,
    ("pass_manager", run),  # Fail
    ("dialects/affine", run_with_insertion_point_v2),  # Pass
    ("dialects/func", run_with_insertion_point_v2),  # Pass
    ("dialects/arith_dialect", run),  # Pass
    ("dialects/arith_llvm", run),  # Pass
    ("dialects/async_dialect", run),  # Pass
    ("dialects/builtin", run),  # Pass
    ("dialects/cf", run_with_insertion_point_v4),  # Pass
    ("dialects/complex_dialect", run),  # Pass
    ("dialects/func", run_with_insertion_point_v2),  # Pass
    ("dialects/index_dialect", run_with_insertion_point),  # Pass
    ("dialects/llvm", run_with_insertion_point_v2),  # Pass
    ("dialects/math_dialect", run),  # Pass
    ("dialects/memref", run),  # Fail
    ("dialects/ml_program", run_with_insertion_point_v2),  # Pass
    ("dialects/nvgpu", run_with_insertion_point_v2),  # Pass
    ("dialects/nvvm", run_with_insertion_point_v2),  # Pass
    ("dialects/ods_helpers", run),  # Pass
    ("dialects/openmp_ops", run_with_insertion_point_v2),  # Pass
    ("dialects/pdl_ops", run_with_insertion_point_v2),  # Pass
    # ("dialects/python_test", run),  # TODO: Need to pass pybind11 or nanobind argv
    ("dialects/quant", run),  # Pass
    ("dialects/rocdl", run_with_insertion_point_v2),  # Pass
    ("dialects/scf", run_with_insertion_point_v2),  # Pass
    ("dialects/shape", run),  # Pass
    ("dialects/spirv_dialect", run),  # Pass
    ("dialects/tensor", run),  # Pass
    # ("dialects/tosa", ),  # Nothing to test
    ("dialects/transform_bufferization_ext", run_with_insertion_point_v2),  # Pass
    # ("dialects/transform_extras", ),  # Needs a more complicated execution schema
    ("dialects/transform_gpu_ext", run_transform_tensor_ext),  # Pass
    (
        "dialects/transform_interpreter",
        run_with_context_and_location,
        ["print_", "transform_options", "failed", "include"],
    ),  # Fail
    (
        "dialects/transform_loop_ext",
        run_with_insertion_point_v2,
        ["loopOutline"],
    ),  # Pass
    ("dialects/transform_memref_ext", run_with_insertion_point_v2),  # Pass
    ("dialects/transform_nvgpu_ext", run_with_insertion_point_v2),  # Pass
    ("dialects/transform_sparse_tensor_ext", run_transform_tensor_ext),  # Pass
    ("dialects/transform_structured_ext", run_transform_structured_ext),  # Pass
    ("dialects/transform_tensor_ext", run_transform_tensor_ext),  # Pass
    (
        "dialects/transform_vector_ext",
        run_apply_patterns,
        ["configurable_patterns"],
    ),  # Pass
    ("dialects/transform", run_with_insertion_point_v3),  # Pass
    ("dialects/vector", run_with_context_and_location),  # Pass
    ("dialects/gpu/dialect", run_with_context_and_location),  # Pass
    ("dialects/gpu/module-to-binary-nvvm", run_with_context_and_location),  # Pass
    ("dialects/gpu/module-to-binary-rocdl", run_with_context_and_location),  # Fail
    ("dialects/linalg/ops", run),  # Pass
    # TO ADD: No proper tests in this dialects/linalg/opsdsl/*
    # ("dialects/linalg/opsdsl/*", ...),  #
    ("dialects/sparse_tensor/dialect", run),  # Pass
    ("dialects/sparse_tensor/passes", run),  # Pass
    ("integration/dialects/pdl", run_construct_and_print_in_module),  # Pass
    ("integration/dialects/transform", run_construct_and_print_in_module),  # Pass
    ("integration/dialects/linalg/opsrun", run),  # Fail
    ("ir/affine_expr", run),  # Pass
    ("ir/affine_map", run),  # Pass
    ("ir/array_attributes", run),  # Pass
    ("ir/attributes", run),  # Pass
    ("ir/blocks", run),  # Pass
    ("ir/builtin_types", run),  # Pass
    ("ir/context_managers", run),  # Pass
    ("ir/debug", run),  # Fail
    ("ir/diagnostic_handler", run),  # Fail
    ("ir/dialects", run),  # Fail
    ("ir/exception", run),  # Pass
    ("ir/insertion_point", run),  # Pass
    ("ir/integer_set", run),  # Pass
    ("ir/location", run),  # Pass
    ("ir/module", run),  # Fail
    ("ir/operation", run),  # Pass
    ("ir/symbol_table", run),  # Pass
    ("ir/value", run),  # Pass
]

TESTS_TO_SKIP = [
    "test_execution_engine__testNanoTime_multi_threaded",  # testNanoTime can't run in multiple threads, even with GIL
    "test_execution_engine__testSharedLibLoad_multi_threaded",  # testSharedLibLoad can't run in multiple threads, even with GIL
    "test_dialects_arith_dialect__testArithValue_multi_threaded",  # RuntimeError: Value caster is already registered: <class 'dialects/arith_dialect.testArithValue.<locals>.ArithValue'>, even with GIL
    "test_ir_dialects__testAppendPrefixSearchPath_multi_threaded",  # PyGlobals::setDialectSearchPrefixes is not thread-safe, even with GIL. Strange usage of static PyGlobals vs python exposed _cext.globals
    "test_ir_value__testValueCasters_multi_threaded_multi_threaded",  # RuntimeError: Value caster is already registered: <function testValueCasters.<locals>.dont_cast_int, even with GIL
    "test_ir_operation_testKnownOpView_multi_threaded_multi_threaded",  # uses register_operation method in the test
    "test_dialects_transform_structured_ext__testMatchInterfaceEnumReplaceAttributeBuilder_multi_threaded",  # uses register_attribute_builder method in the test
]

TESTS_TO_XFAIL = [
    # execution_engine tests, ctypes related data-races, may be false-positive as libffi was not instrumented with TSAN
    "test_execution_engine__testBF16Memref_multi_threaded",
    "test_execution_engine__testBasicCallback_multi_threaded",
    "test_execution_engine__testCapsule_multi_threaded",
    "test_execution_engine__testComplexMemrefAdd_multi_threaded",
    "test_execution_engine__testComplexUnrankedMemrefAdd_multi_threaded",
    "test_execution_engine__testDumpToObjectFile_multi_threaded",
    "test_execution_engine__testDynamicMemrefAdd2D_multi_threaded",
    "test_execution_engine__testF16MemrefAdd_multi_threaded",
    "test_execution_engine__testF8E5M2Memref_multi_threaded",
    "test_execution_engine__testInvalidModule_multi_threaded",
    "test_execution_engine__testInvokeFloatAdd_multi_threaded",
    "test_execution_engine__testInvokeVoid_multi_threaded",
    "test_execution_engine__testMemrefAdd_multi_threaded",
    "test_execution_engine__testRankedMemRefCallback_multi_threaded",
    "test_execution_engine__testRankedMemRefWithOffsetCallback_multi_threaded",
    "test_execution_engine__testUnrankedMemRefCallback_multi_threaded",
    "test_execution_engine__testUnrankedMemRefWithOffsetCallback_multi_threaded",
    # pass_manager tests
    "test_pass_manager__testPrintIrAfterAll_multi_threaded",  # IRPrinterInstrumentation::runAfterPass is not thread-safe
    "test_pass_manager__testPrintIrBeforeAndAfterAll_multi_threaded",  # IRPrinterInstrumentation::runBeforePass is not thread-safe
    "test_pass_manager__testPrintIrLargeLimitElements_multi_threaded",  # IRPrinterInstrumentation::runAfterPass is not thread-safe
    "test_pass_manager__testPrintIrTree_multi_threaded",  # IRPrinterInstrumentation::runAfterPass is not thread-safe
    "test_pass_manager__testRunPipeline_multi_threaded",  # PrintOpStatsPass::printSummary is not thread-safe
    # dialects tests
    "test_dialects_memref__testSubViewOpInferReturnTypeExtensiveSlicing_multi_threaded",  # Related to ctypes data races
    "test_dialects_transform_interpreter__include_multi_threaded",  # mlir::transform::PrintOp::apply(mlir::transform::TransformRewriter...) is not thread-safe
    "test_dialects_transform_interpreter__print_other_multi_threaded",  # Fatal Python error: Aborted or mlir::transform::PrintOp::apply(mlir::transform::TransformRewriter...) is not thread-safe
    "test_dialects_transform_interpreter__print_self_multi_threaded",  # mlir::transform::PrintOp::apply(mlir::transform::TransformRewriter...) is not thread-safe
    "test_dialects_transform_interpreter__transform_options_multi_threaded",  # mlir::transform::PrintOp::apply(mlir::transform::TransformRewriter...) is not thread-safe
    "test_dialects_gpu_module-to-binary-rocdl__testGPUToASMBin_multi_threaded",  # Due to global llvm-project/llvm/lib/Target/AMDGPU/GCNSchedStrategy.cpp::GCNTrackers variable mutation
    "test_dialects_gpu_module-to-binary-nvvm__testGPUToASMBin_multi_threaded",
    "test_dialects_gpu_module-to-binary-nvvm__testGPUToLLVMBin_multi_threaded",
    "test_dialects_gpu_module-to-binary-rocdl__testGPUToLLVMBin_multi_threaded",
    # integration tests
    "test_integration_dialects_linalg_opsrun__test_elemwise_builtin_multi_threaded",  # Related to ctypes data races
    "test_integration_dialects_linalg_opsrun__test_elemwise_generic_multi_threaded",  # Related to ctypes data races
    "test_integration_dialects_linalg_opsrun__test_fill_builtin_multi_threaded",  # ctypes
    "test_integration_dialects_linalg_opsrun__test_fill_generic_multi_threaded",  # ctypes
    "test_integration_dialects_linalg_opsrun__test_fill_rng_builtin_multi_threaded", # ctypes
    "test_integration_dialects_linalg_opsrun__test_fill_rng_generic_multi_threaded", # ctypes
    "test_integration_dialects_linalg_opsrun__test_max_pooling_builtin_multi_threaded", # ctypes
    "test_integration_dialects_linalg_opsrun__test_max_pooling_generic_multi_threaded", # ctypes
    "test_integration_dialects_linalg_opsrun__test_min_pooling_builtin_multi_threaded", # ctypes
    "test_integration_dialects_linalg_opsrun__test_min_pooling_generic_multi_threaded", # ctypes
    # IR tests
    "test_ir_diagnostic_handler__testDiagnosticCallbackException_multi_threaded",  # mlirEmitError is not thread-safe
    "test_ir_module__testParseSuccess_multi_threaded",  #  mlirOperationDump is not thread-safe

    # https://github.com/python/cpython/issues/128013
    "test_ir_module__testRoundtripUnicode_multi_threaded",
    "test_ir_module__testRoundtripBinary_multi_threaded",
]


def add_existing_tests(test_modules, test_prefix: str = "_original_test"):
    def decorator(test_cls):
        this_folder = Path(__file__).parent.absolute()
        test_cls.output_folder = tempfile.TemporaryDirectory()
        output_folder = Path(test_cls.output_folder.name)

        for test_mod_info in test_modules:
            assert isinstance(test_mod_info, tuple) and len(test_mod_info) in (2, 3)
            if len(test_mod_info) == 2:
                test_module_name, exec_fn = test_mod_info
                test_pattern = None
            else:
                test_module_name, exec_fn, test_pattern = test_mod_info

            src_filepath = this_folder / f"{test_module_name}.py"
            dst_filepath = (output_folder / f"{test_module_name}.py").absolute()
            if not dst_filepath.parent.exists():
                dst_filepath.parent.mkdir(parents=True)
            copy_and_update(src_filepath, dst_filepath)
            test_mod = import_from_path(test_module_name, dst_filepath)
            for attr_name in dir(test_mod):
                is_test_fn = test_pattern is None and attr_name.startswith("test")
                is_test_fn |= test_pattern is not None and any(
                    [p in attr_name for p in test_pattern]
                )
                if is_test_fn:
                    obj = getattr(test_mod, attr_name)
                    if callable(obj):
                        test_name = f"{test_prefix}_{test_module_name.replace('/', '_')}__{attr_name}"

                        def wrapped_test_fn(
                            self, *args, __test_fn__=obj, __exec_fn__=exec_fn, **kwargs
                        ):
                            __exec_fn__(__test_fn__)

                        setattr(test_cls, test_name, wrapped_test_fn)
        return test_cls

    return decorator


def multi_threaded(
    num_workers: int,
    num_runs: int = 5,
    skip_tests: Optional[list[str]] = None,
    xfail_tests: Optional[list[str]] = None,
    test_prefix: str = "_original_test",
    multithreaded_test_postfix: str = "_multi_threaded",
):
    """Decorator that runs a test in a multi-threaded environment."""

    def decorator(test_cls):
        for name, test_fn in test_cls.__dict__.copy().items():
            if not (name.startswith(test_prefix) and callable(test_fn)):
                continue

            name = f"test{name[len(test_prefix):]}"
            if skip_tests is not None:
                if any(
                    test_name.replace(multithreaded_test_postfix, "") in name
                    for test_name in skip_tests
                ):
                    continue

            def multi_threaded_test_fn(
                self, capfd, *args, __test_fn__=test_fn, **kwargs
            ):
                barrier = threading.Barrier(num_workers)

                def closure():
                    barrier.wait()
                    for _ in range(num_runs):
                        __test_fn__(self, *args, **kwargs)

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    futures = []
                    for _ in range(num_workers):
                        futures.append(executor.submit(closure))
                    # We should call future.result() to re-raise an exception if test has
                    # failed
                    assert len(list(f.result() for f in futures)) == num_workers

                gc.collect()
                assert Context._get_live_count() == 0

                captured = capfd.readouterr()
                if len(captured.err) > 0:
                    if "ThreadSanitizer" in captured.err:
                        raise RuntimeError(
                            f"ThreadSanitizer reported warnings:\n{captured.err}"
                        )
                    else:
                        pass
                        # There are tests that write to stderr, we should ignore them
                        # raise RuntimeError(f"Other error:\n{captured.err}")

            test_new_name = f"{name}{multithreaded_test_postfix}"
            if xfail_tests is not None and test_new_name in xfail_tests:
                multi_threaded_test_fn = pytest.mark.xfail(multi_threaded_test_fn)

            setattr(test_cls, test_new_name, multi_threaded_test_fn)

        return test_cls

    return decorator


@multi_threaded(
    num_workers=8,
    num_runs=20,
    skip_tests=TESTS_TO_SKIP,
    xfail_tests=TESTS_TO_XFAIL,
)
@add_existing_tests(test_modules=TEST_MODULES, test_prefix="_original_test")
class TestAllMultiThreaded:
    @pytest.fixture(scope="class")
    def teardown(self):
        if hasattr(self, "output_folder"):
            self.output_folder.cleanup()

    def _original_test_create_context(self):
        with Context() as ctx:
            print(ctx._get_live_count())
            print(ctx._get_live_module_count())
            print(ctx._get_live_operation_count())
            print(ctx._get_live_operation_objects())
            print(ctx._get_context_again() is ctx)
            print(ctx._clear_live_operations())

    def _original_test_create_module_with_consts(self):
        py_values = [123, 234, 345]
        with Context() as ctx:
            module = Module.create(loc=Location.file("foo.txt", 0, 0))

            dtype = IntegerType.get_signless(64)
            with InsertionPoint(module.body), Location.name("a"):
                arith.constant(dtype, py_values[0])

            with InsertionPoint(module.body), Location.name("b"):
                arith.constant(dtype, py_values[1])

            with InsertionPoint(module.body), Location.name("c"):
                arith.constant(dtype, py_values[2])
