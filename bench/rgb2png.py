import sys, zlib, struct
src, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
raw = open(src, 'rb').read()
rows = b''.join(b'\x00' + raw[y*w*3:(y+1)*w*3] for y in range(h))
def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows, 6)) + chunk(b'IEND', b'')
open(src[:-4] + '.png', 'wb').write(png)
