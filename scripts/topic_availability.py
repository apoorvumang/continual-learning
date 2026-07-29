"""Score candidate SDF topics by how much real coverage we can actually retrieve.

A topic is only usable if (a) the model provably doesn't know it -- read off the committed
baseline -- and (b) we can pull enough real reporting to build a solid universe context.
This measures (b): distinct URLs and distinct domains, restricted to the weeks after the
event so we get contemporaneous reporting rather than later retrospectives.

    python scripts/topic_availability.py
"""

from __future__ import annotations

import datetime as dt
import json
from urllib.parse import urlparse

from keenable import search

# Candidate "really big events", all labelled `incorrect` on the baseline direct probe.
# event_id ties each back to the benchmark row it is meant to fix.
TOPICS = [
    ("charlie-kirk", "2025-09-10", ["Charlie Kirk assassination Utah Valley University",
                                    "Charlie Kirk shot killed Turning Point USA",
                                    "Tyler Robinson charged Charlie Kirk shooting"]),
    ("khamenei", "2026-02-28", ["Ali Khamenei killed Israeli strike",
                                "Iran Supreme Leader Khamenei death",
                                "Khamenei successor Assembly of Experts"]),
    ("maduro", "2026-01-03", ["Nicolas Maduro captured US forces Venezuela",
                              "Maduro seized Caracas operation",
                              "Venezuela after Maduro capture"]),
    ("takaichi", "2025-10-21", ["Sanae Takaichi elected Prime Minister Japan",
                                "Takaichi first female Japanese prime minister",
                                "Ishiba resignation successor LDP leadership"]),
    ("ozzy", "2025-07-22", ["Ozzy Osbourne dies Black Sabbath",
                            "Ozzy Osbourne death tributes",
                            "Ozzy Osbourne final concert Back to the Beginning"]),
    ("cheney", "2025-11-03", ["Dick Cheney dies former vice president",
                              "Dick Cheney death obituary",
                              "Cheney legacy reaction death"]),
    ("keaton", "2025-10-11", ["Diane Keaton dies actress",
                              "Diane Keaton death Santa Monica",
                              "Diane Keaton tributes Annie Hall"]),
    ("goodall", "2025-10-01", ["Jane Goodall dies primatologist",
                               "Jane Goodall death natural causes",
                               "Jane Goodall legacy chimpanzees tributes"]),
    ("mamdani", "2025-11-04", ["Zohran Mamdani wins New York City mayoral election",
                               "Mamdani defeats Cuomo mayor",
                               "Mamdani mayor victory speech"]),
    ("apple-ceo", "2026-04-20", ["Tim Cook steps down Apple CEO",
                                 "Apple CEO succession announcement",
                                 "new Apple chief executive named"]),
]

WINDOW_DAYS = 45


def main():
    rows = []
    for name, date, queries in TOPICS:
        d0 = dt.date.fromisoformat(date)
        after = (d0 - dt.timedelta(days=2)).isoformat()
        before = (d0 + dt.timedelta(days=WINDOW_DAYS)).isoformat()
        urls, domains, dated = {}, set(), 0
        for q in queries:
            for r in search(q, published_after=after, published_before=before):
                u = r["url"]
                if u not in urls:
                    urls[u] = r
                    domains.add(urlparse(u).netloc.replace("www.", ""))
                    if r.get("published_at"):
                        dated += 1
        rows.append({"topic": name, "event_date": date, "queries": len(queries),
                     "urls": len(urls), "domains": len(domains), "with_date": dated,
                     "sample_domains": sorted(domains)[:6]})
        print(json.dumps(rows[-1]), flush=True)

    print("\n{:14s} {:>5s} {:>8s} {:>8s}".format("topic", "urls", "domains", "dated"))
    for r in sorted(rows, key=lambda x: -x["urls"]):
        print("{:14s} {:5d} {:8d} {:8d}".format(
            r["topic"], r["urls"], r["domains"], r["with_date"]))


if __name__ == "__main__":
    main()
