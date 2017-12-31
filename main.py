# coding=utf-8
import os
import commands
import subprocess
import tkMessageBox
import time
import pylab
from io import BytesIO
from PIL import Image, ImageDraw
import threading
import math

ADB_BIN = 'adb'

IN_TESTING = True
SRC_WIDTH = 1080
SRC_HEIGHT = 2160
PT_SIZE = 10
SLEEP_DURATION = 1.0
VIEW_DPI = 180

INITIAL_B = 80
INITIAL_W = 1.262
# CALIBRATE_W = 1.4846665163
# CALIBRATE_B = 5.57542237468
CALIBRATE_W = None
CALIBRATE_B = None

# 跳跃小人的模型，很简单，检测两个制定颜色的点，以后可以精确点
# CHARACTER_MODEL = (
#    (0, 0, 55, 56, 97),
#    (19, 0, 55, 56, 97)
# )

CHARACTER_MODEL = (
    (0, 0, 57, 57, 99),
    (1, 0, 57, 57, 99),
    (2, 0, 57, 57, 99),
    (3, 0, 57, 57, 99),
    (4, 0, 57, 57, 99),
    (1, 1, 57, 57, 99),
    (2, 1, 57, 57, 99),
    (3, 1, 57, 57, 99),
    (4, 1, 57, 57, 99),
    (4, 2, 57, 57, 99),
)


# 执行命令行
def cmd(line, out_is_binary=False):
    cmdline = line if isinstance(line, str) else ' '.join(line)
    if out_is_binary:
        with os.tmpfile() as stdout:
            status = subprocess.call(line, stdout=stdout)
            stdout.seek(0)
            buf = BytesIO()
            buf.write(stdout.read())
            output = buf
    else:
        status, output = commands.getstatusoutput(line)
    output_log = output if not out_is_binary else '<binary data>'
    print('"%s" returned %s, and says:%s%s' % (cmdline, status, os.linesep, output_log))
    return status, output


# 检测adb
def detect_adb():
    # 检测adb
    status, output = cmd('%s devices' % ADB_BIN)
    if status:
        raise EnvironmentError('找不到USB调试客户端adb可执行文件 ADB_BIN: %s' % ADB_BIN)
    lines = output.splitlines(False)
    lines.pop(0)
    # 检测设备数
    devices = [line.split('\t') for line in lines]
    if len(devices) > 1:
        raise EnvironmentError('已连接多个Android设备，请移除其他设备连接，只保留游戏设备连接')
    if len(devices) < 1:
        raise EnvironmentError('未找到您的Android设备，请确认设备已经开启USB调试并连接到电脑')

    # 检测是否支持截图
    _, output = cmd('%s shell screencap -h' % ADB_BIN)
    if output.find('usage:') < 0:
        raise EnvironmentError('您的Android设备不支持通过USB调试截屏')
    # 检测 模拟事件是否支持
    _, output = cmd('%s shell input' % ADB_BIN)
    usage_found = output.find('Usage:') > -1
    swipe_found = output.find('swipe') > -1
    if not usage_found:
        raise EnvironmentError('您的Android设备不支持模拟输入')
    if not swipe_found:
        raise EnvironmentError('您的Android设备不支持模拟长按')

    # 检测长按功能是否授权
    status, output = cmd('%s shell input swipe -1 -1 -1 -1 0' % ADB_BIN)
    if status > 0:
        raise EnvironmentError('您的Android设备未授权通过USB调试模拟输入，请确认在开发者设置中开启允许模拟点击(为了您的设备安全，使用结束后建议关闭模拟点击开关)')

    None if IN_TESTING else tkMessageBox.showwarning('安全建议', """您的Android设备已经开启通过USB调试模拟点击，这是运行本程序必需条件之一，允许USB\
    调试模拟点击存在安全风险，如无其他要求，建议您使用完毕后关闭允许USB调试模拟点击""")
    pass


def capture():
    status, output = cmd([ADB_BIN, 'exec-out', 'screencap', '-p'], out_is_binary=True)
    if status:
        raise RuntimeError('通过USB调试截屏失败')
    fp = output
    return Image.open(fp)


# 在图像中查找小人
def find_character(image, limit=(0, 0, 0, 0)):
    model = CHARACTER_MODEL
    data = image.load()
    min_x, min_y, max_x, max_y, cx, cy = get_model_info(model, limit)
    for y in xrange(min_y, max_y):
        for x in xrange(min_x, max_x):
            match_count = 0
            for p in model:
                X, Y, R, G, B = p
                r, g, b, _ = data[x + X, y + Y]
                if r == R and g == G and b == B:
                    match_count = match_count + 1
                else:
                    break
            if match_count == len(model):
                return x + cx, y + cy
    return -1, -1


# 检测模型查找边界和模型中心
def get_model_info(model, limit=(0, 0, 0, 0)):
    min_x, min_y, max_x, max_y = limit
    w, h = max_x, max_y
    nx, ny, px, py = (0, 0, 0, 0)
    for x, y, _, _, _ in model:
        min_x, nx = (max(min_x, -x), min(nx, x)) if x < 0 else (min_x, nx)
        max_x, px = (min(max_x, w - x), max(px, x)) if x > 0 else (max_x, px)
        min_y, ny = (max(min_y, -y), min(ny, y)) if y < 0 else (min_y, ny)
        max_y, py = (min(max_y, h - y), max(py, y)) if x > 0 else (max_y, py)
    return min_x, min_y, max_x, max_y, (px - nx) / 2 + nx, (py - ny) / 2 + ny


def distance_to_duration(distance):
    # return int (1.1563 * distance + 164)
    if CALIBRATE_W is not None:
        return int(CALIBRATE_W * distance + CALIBRATE_B)
    return int(INITIAL_W * distance + INITIAL_B)
    # return int(1.291 * distance + 29.695)


def jump(duration):
    cmd('adb shell input swipe 200 200 200 200 %s' % duration)


def find_calibrate_points(image, center=(0, 0)):
    return find_calibrate_points_a(image, center)

def find_calibrate_points_b(image, center=(0, 0)):
    return ''

def find_calibrate_points_a(image, center=(0, 0)):
    x, y = center
    points = []
    offset_x = 60
    offset_y = 60
    step_x = 3
    step_y = 3
    with image.convert('L') as im:
        data = im.load()
        w, h = im.size
        xs = [-x if x < offset_x else -offset_x, min(w - x, offset_x)]

        for cy in range(offset_y, offset_y * 2, step_y):
            for cx in xs:
                # print (cx + x, cy + y)
                v = data[cx + x, cy + y]
                points.append((x, y, cx, cy, v))
        mid = offset_y + offset_y / 2
        for cx in range(xs[0] + 1, xs[1] - 1, step_x):
            v = data[cx + x, y + mid]
            points.append((x, y, cx, mid, v))

    return points


def find_matched_points(image, last_points, start_point=(0, 0), end_point=None):
    w, h = image.size
    if end_point is None:
        end_point = (w, h)
    min_count = len(last_points) * 0.8
    max_miss = len(last_points) - min_count
    xset = [p[2] for p in last_points]
    min_offset_x = min(xset)
    max_offset_x = max(xset)
    yset = [p[3] for p in last_points]
    min_offset_y = min(yset)
    max_offset_y = max(yset)

    points = []
    with image.convert('L') as im:
        data = im.load()
        for y in xrange(start_point[1] - min(0, min_offset_y), end_point[1] - max(0, max_offset_y)):
            for x in xrange(start_point[0] - min(0, min_offset_x), end_point[0] - max(0, max_offset_x)):
                match_count = 0
                miss_count = 1
                for _, _, offset_x, offset_y, tv in last_points:
                    if miss_count > max_miss:
                        break
                    tx = x + offset_x
                    ty = y + offset_y
                    if tx < 0 or tx > end_point[0] or tx >= w:
                        miss_count += 1
                        continue
                    if ty < 0 or ty > end_point[1] or ty >= h:
                        miss_count += 1
                        continue
                    # print(tx, ty)
                    v = data[tx, ty]
                    match_count += 0 if v != tv else 1
                    miss_count += 1 if v != tv else 0
                if match_count > min_count:
                    points.append((x, y, match_count))
    points.sort(lambda x1, x2: x1[2] - x2[2], reverse=True)
    return points


def calibrate_vars(params):
    global CALIBRATE_W
    global CALIBRATE_B
    # duration 允许的最大误差
    max_lost = 16
    (x1, y1), (x2, y2) = params[:2]
    # 求常数项
    scale = x2 * 1.0 / x1
    b = (y2 - y1 * scale)
    w = (y2 - b) / x2
    # 验证
    delta = [abs(x * w + b - y) for (x, y) in params[2:]]
    print('delta with %s, %s : %s' % (w, b, delta))
    ret = max(delta) < max_lost
    if ret:
        CALIBRATE_B = b
        CALIBRATE_W = w
    return ret


# 校准
def calibrate():
    try:
        detect_adb()
        # time.sleep(1)
    except EnvironmentError, e:
        tkMessageBox.showerror('无法运行', e.message)

    fig = pylab.figure(dpi=220)
    # 特征点
    points = []
    params = []
    duration = 0
    calibrate_success = False
    # 游戏循环
    while True:
        with capture() as im:
            fig.clear()
            w, h = im.size
            limit = (w / 8, h / 2 - h / 8, w - w / 8, h / 2 + h / 4)
            x, y = find_character(im, limit)
            image = pylab.array(im)
            pylab.imshow(image)
            # 绘制上次的采样
            if len(points) > 0 and not calibrate_success:
                matches = find_matched_points(im, points, (0, y + 80), (w, y + 700))
                top_match = matches[:5]
                if len(top_match) > 1:
                    # 加权平均值
                    top_match_w = sum([p[2] for p in top_match])
                    top_match_x = sum([p[0] * p[2] for p in top_match]) / top_match_w
                    top_match_y = sum([p[1] * p[2] for p in top_match]) / top_match_w
                    pylab.plot([top_match_x], [top_match_y], 'b.')
                    # 将参数加入到params里面
                    d = math.sqrt(math.pow(top_match_x - x, 2) + math.pow(top_match_y - y, 2))
                    params.append((d, duration))
                    print(d, duration)
                    if len(params) >= 5:
                        calibrate_success = calibrate_vars(params)
                        params = []
            if not calibrate_success:
                points = find_calibrate_points(im, (x, y))
                pylab.plot([p[0] + p[2] for p in points], [p[1] + p[3] for p in points], 'c|')
            pylab.plot([x], [y], 'g+')
            pylab.draw()
            pylab.show(block=False)
            tx, ty = pylab.ginput(1, timeout=0)[0]
            distance = math.sqrt(math.pow(x - tx, 2) + math.pow(y - ty, 2))
            duration = distance_to_duration(distance)
            jump(duration)
            time.sleep(SLEEP_DURATION + duration / 1000.0)


def main():
    try:
        detect_adb()
        # time.sleep(1)
    except EnvironmentError, e:
        tkMessageBox.showerror('无法运行', e.message)

    pylab.figure(dpi=VIEW_DPI)
    # 游戏循环
    while True:
        with capture() as im:
            w, h = im.size
            limit = (w / 8, h / 2 - h / 8, w - w / 8, h / 2 + h / 4)
            x, y = find_character(im, limit)
            # x, y 做下微调
            # x += 5
            draw = ImageDraw.Draw(im)
            draw.ellipse((x - PT_SIZE, y - PT_SIZE, x + PT_SIZE, y + PT_SIZE), fill='red')
            del draw
            image = pylab.array(im)
            pylab.imshow(image)
            pylab.draw()
            pylab.show(block=False)
            tx, ty = pylab.ginput(1, timeout=0)[0]
            distance = math.sqrt(math.pow(x - tx, 2) + math.pow(y - ty, 2))
            duration = distance_to_duration(distance)
            jump(duration)
            time.sleep(SLEEP_DURATION + duration / 1000.0)


# main()
calibrate()
