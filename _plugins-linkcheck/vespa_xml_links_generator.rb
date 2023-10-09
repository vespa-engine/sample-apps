# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

require 'nokogiri'

module Jekyll

    # This generator creates links-to-check.html with links extracted from comments in named XML files
    # This file can be moved into the _site directory for link check subsequently

    class VespaXMLLinksGenerator < Jekyll::Generator
        priority :lowest

        def generate(site)
            all_links = []
            site.pages.each do |page|
                if ["services.xml", "hosts.xml", "deployment.xml"].include?(page.name)
                    extract_links(page).each do |link|
                        all_links.push(link)
                    end
                end
                if page.name.include?".java"
                    extract_links_javadoc(page).each do |link|
                        all_links.push(link)
                    end
                end
                if page.name.include?".sd"
                    extract_links_sd(page).each do |link|
                        all_links.push(link)
                    end
                end
            end
            link_file = "<!DOCTYPE html><html><head><title>links</title></head><body>\n"
            all_links.uniq.each do |link|
                link_file += "<a href='" + link + "'>" + link + "</a>\n"
            end
            link_file += "</body></html>\n"
            File.open("links-to-check.html", "w") { |f| f.write(link_file) }
        end

        def extract_links(page)
            all_urls = []
            doc = Nokogiri::HTML(page.content)
            comments = doc.xpath("//comment()")
            comments.each do |comment|
                urls = comment.to_s.split(/\s+/).find_all { |u| u =~ /^https?:/ }
                urls.each do |url|
                    all_urls.push(url)
                end
            end
            return all_urls
        end

        def extract_links_javadoc(page)
            all_urls = []
            page.content.each_line { |line|
                if line =~ /^\s+\*\s/
                    urls = line.gsub('"', ' ').gsub("'", ' ').split(/\s+/).find_all { |u| u =~ /^https?:/ }
                    urls.each do |url|
                        all_urls.push(url)
                    end
                end
            }
            return all_urls
        end

        def extract_links_sd(page)
            all_urls = []
            page.content.each_line { |line|
                if line =~ /#/
                    urls = line.split(/\s+/).find_all { |u| u =~ /^https?:/ }
                    urls.each do |url|
                        all_urls.push(url)
                    end
                end
            }
            return all_urls
        end

    end

end
