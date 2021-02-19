# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

require 'json'
require 'nokogiri'
require 'kramdown/parser/kramdown'

module Jekyll

    class VespaIndexGenerator < Jekyll::Generator
        priority :lowest

        def generate(site)
            namespace = site.config["search"]["namespace"]
            operations = []
            site.pages.each do |page|
                appname = page.url[1..page.url.index("README.html")-2]  # from /use-case-shopping/README.html to use-case-shopping
                if page.data["index"] == true
                    operations.push({
                        :fields => {
                            :path => page.url,
                            :namespace => namespace,
                            :title => "Vespa Sample Applications: " + appname,
                            :content => extract_text(page)
                        }
                    })
                end
            end

            json = JSON.pretty_generate(operations)
            File.open(namespace + "_index.json", "w") { |f| f.write(json) }
        end

        def extract_text(page)
            ext = page.name[page.name.rindex('.')+1..-1]
            if ext == "md"
                input = Kramdown::Document.new(page.content).to_html
            else
                input = page.content
            end
            doc = Nokogiri::HTML(input)
            doc.search('th,td').each{ |e| e.after "\n" }
            content = doc.xpath("//text()").to_s
            page_text = content.gsub("\r"," ").gsub("\n"," ")
        end

    end

end
