# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

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
                if not page.url.include?("README")
                    next
                end                
                if page.url.include?("-README.html")
                    path = page.url.sub(/README.html/, "README.md")
                else
                    path = page.url[0..page.url.rindex("/")]  # link to repo dir instead of README.md
                end

                title = page.url[0..page.url.rindex("README.html")-1].gsub(/(\/|-)$/, "")
                if title.empty?
                    title = "Vespa Sample Applications"
                else
                    title = "Vespa Sample Applications: " + title
                end

                if page.data["index"] == true

                    operations.push({
                        :put => "id:" + namespace + ":doc::" + namespace + path,
                        :fields => {
                            :path => path,
                            :namespace => namespace,
                            :title => title,
                            :content => extract_text(page),
                            :html => Kramdown::Document.new(page.content).to_html
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
            doc.search('style').each{ |e| e.remove }
            content = doc.xpath("//text()").to_s
            page_text = content.gsub("\r"," ").gsub("\n"," ")
        end

    end

end
