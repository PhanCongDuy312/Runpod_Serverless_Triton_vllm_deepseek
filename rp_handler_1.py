#!/usr/bin/env python3
#
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import asyncio
import json
import sys
import time
import nest_asyncio
nest_asyncio.apply()  # Allow nested event loops

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *
import runpod

# Health polling function
def wait_for_triton(timeout=360):
    """Polls Triton's health endpoint until ready or timeout reached."""
    start_time = time.time()
    url = "http://localhost:1235/v2/health/ready"
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            # Assuming a 200 response indicates Triton is ready.
            if response.status_code == 200:
                print("Triton is ready!")
                return
        except Exception as e:
            pass
        print("Waiting for Triton to be ready...")
        time.sleep(2)
    raise Exception(f"Triton server not ready after {timeout} seconds.")

class LLMClient:
    def __init__(self, flags: argparse.Namespace):
        self._flags = flags
        self._results_dict = {}

    def get_triton_client(self):
        try:
            triton_client = grpcclient.InferenceServerClient(
                url=self._flags.url,
                verbose=self._flags.verbose,
            )
        except Exception as e:
            print("Channel creation failed: " + str(e))
            sys.exit(1)
        return triton_client

    async def async_request_iterator(self, prompts, sampling_parameters, exclude_input_in_output):
        try:
            for iter in range(self._flags.iterations):
                for i, prompt in enumerate(prompts):
                    prompt_id = self._flags.offset + (len(prompts) * iter) + i
                    self._results_dict[str(prompt_id)] = []
                    yield self.create_request(
                        prompt,
                        self._flags.streaming_mode,
                        prompt_id,
                        sampling_parameters,
                        exclude_input_in_output,
                    )
        except Exception as error:
            print(f"Caught an error in the request iterator: {error}")

    async def stream_infer(self, prompts, sampling_parameters, exclude_input_in_output):
        try:
            triton_client = self.get_triton_client()
            # Start streaming inference
            response_iterator = triton_client.stream_infer(
                inputs_iterator=self.async_request_iterator(prompts, sampling_parameters, exclude_input_in_output),
                stream_timeout=self._flags.stream_timeout,
            )
            async for response in response_iterator:
                yield response
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    async def process_stream(self, prompts, sampling_parameters, exclude_input_in_output):
        # Clear previous results
        self._results_dict = {}
        success = True
        async for response in self.stream_infer(prompts, sampling_parameters, exclude_input_in_output):
            result, error = response
            if error:
                print(f"Encountered error while processing: {error}")
                success = False
            else:
                output = result.as_numpy("text_output")
                for i in output:
                    self._results_dict[result.get_response().id].append(i)
        return success

    async def run(self):
        # Sampling parameters for text generation
        sampling_parameters = {
            "temperature": "0.1",
            "top_p": "0.95",
            "max_tokens": "100",
        }
        exclude_input_in_output = self._flags.exclude_inputs_in_outputs
        if self._flags.lora_name is not None:
            sampling_parameters["lora_name"] = self._flags.lora_name

        # Use prompts provided in the event input rather than reading from a file.
        prompts = self._flags.input_prompts
        print("Using provided prompts from event input...")

        success = await self.process_stream(prompts, sampling_parameters, exclude_input_in_output)

        # Build the final result string from the responses.
        final_result = ""
        for id in self._results_dict.keys():
            for result in self._results_dict[id]:
                final_result += result.decode("utf-8")
            final_result += "\n=========\n\n"
        print("Processed results.")

        if self._flags.verbose:
            print(f"\nResults:\n{final_result}")
        if success:
            print("PASS: vLLM example")
        else:
            print("FAIL: vLLM example")
        return final_result

    def run_async(self):
        # Using nest_asyncio allows asyncio.run() even when an event loop is already running.
        return asyncio.run(self.run())

    def create_request(self, prompt, stream, request_id, sampling_parameters, exclude_input_in_output, send_parameters_as_tensor=True):
        inputs = []
        prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
        try:
            inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(prompt_data)
        except Exception as error:
            print(f"Encountered an error during request creation: {error}")

        stream_data = np.array([stream], dtype=bool)
        inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(stream_data)

        if send_parameters_as_tensor:
            sampling_parameters_data = np.array([json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_)
            inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(sampling_parameters_data)

        inputs.append(grpcclient.InferInput("exclude_input_in_output", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(np.array([exclude_input_in_output], dtype=bool))

        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("text_output"))

        return {
            "model_name": self._flags.model,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": str(request_id),
            "parameters": sampling_parameters,
        }

def handler(event):
    """
    Runpod handler that accepts an event with the following structure:
    {
        "input": {
            "prompts": <a single prompt string or a list of prompt strings>,
            "model": <optional model name>,
            "verbose": <bool>,
            "url": <server URL>,
            "stream_timeout": <float>,
            "offset": <int>,
            "iterations": <int>,
            "streaming_mode": <bool>,
            "exclude_inputs_in_outputs": <bool>,
            "lora_name": <optional lora name>
        }
    }
    """
    input_data = event.get("input", {})

    # If a single prompt string is provided, wrap it in a list.
    prompts = input_data.get("prompts", [])
    if isinstance(prompts, str):
        prompts = [prompts]

    flags = argparse.Namespace(
        model=input_data.get("model", "deepseek-r1-distill-qwen-1.5b-awq"),
        verbose=input_data.get("verbose", False),
        url=input_data.get("url", "localhost:8001"),
        stream_timeout=input_data.get("stream_timeout", None),
        offset=input_data.get("offset", 0),
        input_prompts=prompts,
        iterations=input_data.get("iterations", 1),
        streaming_mode=input_data.get("streaming_mode", False),
        exclude_inputs_in_outputs=input_data.get("exclude_inputs_in_outputs", False),
        lora_name=input_data.get("lora_name", None),
    )

    client = LLMClient(flags)
    result = client.run_async()
    print("===============================================")
    print("FINAL result:",result)
    print("===============================================")
    
    return result

if __name__ == '__main__':
    # time.sleep(10)
    runpod.serverless.start({"handler": handler})
    # try:
    #     wait_for_triton(timeout=60)
    # except Exception as e:
    #     print("Triton server failed to start:", e)
    #     sys.exit(1)
    # runpod.serverless.start({"handler": handler})