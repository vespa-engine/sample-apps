# Copyright Vespa.ai. All rights reserved.
require 'performance_test'
require 'app_generator/search_app'
require 'environment'

class GgufEmbeddingPerfTest < PerformanceTest

  def setup
    set_owner("bjorncs")
    set_description("Benchmark feed throughput with gguf embedding model")
  end

  def test_gguf_feed_throughput
    deploy(selfdir + "app")
    start

    # Original file info
    feed_file = 'miracl-te-docs.10k.json.gz'
    remote_file = "https://data.vespa-cloud.com/tests/performance/#{feed_file}"
    local_file = dirs.tmpdir + feed_file

    # Download the original file
    cmd = "wget -O'#{local_file}' '#{remote_file}'"
    puts "Running command #{cmd}"
    result = `#{cmd}`
    puts "Result: #{result}"

    # Create a new file with only the first 100 documents
    reduced_file = dirs.tmpdir + 'miracl-te-docs.100.json'
    puts "Creating reduced file with 100 documents: #{reduced_file}"
    extract_cmd = "gunzip -c '#{local_file}' | jq '.[:100]' > '#{reduced_file}'"
    puts "Running command #{extract_cmd}"
    result = `#{extract_cmd}`
    puts "Result: #{result}"

    # Feed the reduced file
    run_feeder(reduced_file, [], {:numthreads => 64, :timeout => 1200})
  end
end