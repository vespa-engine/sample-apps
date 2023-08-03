
# i-046aacad7834f9514 == Torstein's vm (simple-ec2-instance)
scp -O -A -o ProxyJump=35.170.131.31:2222 ../cloudbeir/feed.jsonl i-046aacad7834f9514:
scp -r -O -A -o ProxyJump=35.170.131.31:2222 ../application-package i-046aacad7834f9514:
